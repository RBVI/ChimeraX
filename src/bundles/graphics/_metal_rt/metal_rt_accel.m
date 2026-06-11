/*
 * Metal ray tracing: acceleration structures, RTAO, and path tracer.
 *
 * Adapted from _metal/metal_raytracing.m to work with a MetalRTContext
 * (device/queue exported from MoltenVK) rather than a full MetalContext.
 * Dispatch functions create their own command buffers and synchronize
 * with the Vulkan renderer via MTLSharedEvent.
 */

#include <Python.h>
#include "metal_rt_internal.h"
#include "vulkan_internal.h"

// ---------------------------------------------------------------------------
//  Helper: get MetalRTContext* from Python keyword arg "context".
// ---------------------------------------------------------------------------
static MetalRTContext* get_rt_context(PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    static const char* kwlist[] = {"context", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &ctx_ptr))
        return nullptr;
    return reinterpret_cast<MetalRTContext*>(ctx_ptr);
}

// ---------------------------------------------------------------------------
//  Helper: validate that a Python buffer really holds `count` elements of
//  `stride` bytes. newBufferWithBytes copies count*stride bytes; a short
//  buffer (shape/dtype mismatch on the Python side) would read past the
//  end of the numpy allocation — a segfault with large geometry. Raises
//  ValueError naming the offending geometry instead.
// ---------------------------------------------------------------------------
static bool check_geom_buffer(const Py_buffer* buf, long count,
                              size_t stride, const char* what) {
    if (count < 0) {
        PyErr_Format(PyExc_ValueError, "%s count is negative (%ld)", what, count);
        return false;
    }
    if ((size_t)buf->len < (size_t)count * stride) {
        PyErr_Format(PyExc_ValueError,
                     "%s buffer too small: %zd bytes for %ld elements of %zu bytes",
                     what, buf->len, count, stride);
        return false;
    }
    return true;
}

// Helper: validate the context and its exported Metal objects before use.
static bool check_rt_context(MetalRTContext* ctx) {
    if (!ctx) {
        PyErr_SetString(PyExc_ValueError, "Metal RT context is null");
        return false;
    }
    if (!ctx->device || !ctx->commandQueue) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Metal RT context has no device/queue "
                        "(was the Vulkan context recreated?)");
        return false;
    }
    return true;
}

// Cached RTAO pipeline state.
static id<MTLComputePipelineState> s_rtao_pipeline = nil;
static id<MTLFunction> s_rtao_isect_func = nil;
static id<MTLIntersectionFunctionTable> s_rtao_ift = nil;

// Cached path tracer pipeline state.
static id<MTLComputePipelineState> s_pt_pipeline = nil;
static id<MTLIntersectionFunctionTable> s_pt_ift = nil;

// ===========================================================================
//  Acceleration structure building
// ===========================================================================

extern "C"
PyObject* metal_rt_build_sphere_accel(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    Py_buffer sphere_buf;
    int sphere_count;
    static const char* kwlist[] = {"context", "sphere_data", "sphere_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ny*i"), (char**)kwlist,
            &ctx_ptr, &sphere_buf, &sphere_count))
        return NULL;

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    if (!check_rt_context(ctx) ||
        !check_geom_buffer(&sphere_buf, sphere_count, sizeof(float) * 4, "sphere")) {
        PyBuffer_Release(&sphere_buf);
        return NULL;
    }
    id<MTLDevice> device = ctx->device;

    id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:sphere_buf.buf
                                          length:(NSUInteger)sphere_count * sizeof(float) * 4
                                         options:MTLResourceStorageModeShared];
    PyBuffer_Release(&sphere_buf);
    if (!sphereBuffer) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate sphere MTLBuffer");
        return NULL;
    }

    MTLAxisAlignedBoundingBox *aabbs = (MTLAxisAlignedBoundingBox *)malloc(
        sphere_count * sizeof(MTLAxisAlignedBoundingBox));
    float *spheres = (float *)sphereBuffer.contents;
    for (int i = 0; i < sphere_count; i++) {
        float x = spheres[i*4+0], y = spheres[i*4+1], z = spheres[i*4+2], r = spheres[i*4+3];
        aabbs[i].min = MTLPackedFloat3Make(x-r, y-r, z-r);
        aabbs[i].max = MTLPackedFloat3Make(x+r, y+r, z+r);
    }

    id<MTLBuffer> aabbBuffer = [device newBufferWithBytes:aabbs
                                         length:sphere_count * sizeof(MTLAxisAlignedBoundingBox)
                                        options:MTLResourceStorageModeShared];
    free(aabbs);

    MTLAccelerationStructureBoundingBoxGeometryDescriptor *geomDesc =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    geomDesc.boundingBoxBuffer = aabbBuffer;
    geomDesc.boundingBoxCount = sphere_count;
    geomDesc.primitiveDataBuffer = sphereBuffer;
    geomDesc.primitiveDataStride = sizeof(float) * 4;
    geomDesc.primitiveDataElementSize = sizeof(float) * 4;
    geomDesc.intersectionFunctionTableOffset = 0;

    MTLPrimitiveAccelerationStructureDescriptor *asDesc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    asDesc.geometryDescriptors = @[geomDesc];

    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:asDesc];
    id<MTLAccelerationStructure> accelStruct =
        [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratchBuffer =
        [device newBufferWithLength:sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];

    id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> encoder =
        [cmdBuf accelerationStructureCommandEncoder];
    [encoder buildAccelerationStructure:accelStruct
                             descriptor:asDesc
                          scratchBuffer:scratchBuffer
                    scratchBufferOffset:0];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    long as_ptr = (long)(__bridge_retained void *)accelStruct;
    long sb_ptr = (long)(__bridge_retained void *)sphereBuffer;
    return Py_BuildValue("ll", as_ptr, sb_ptr);
}

extern "C"
PyObject* metal_rt_build_mixed_accel(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    Py_buffer sphere_buf, cyl_buf;
    int sphere_count, cyl_count;
    static const char* kwlist[] = {"context", "sphere_data", "sphere_count",
                                   "cylinder_data", "cylinder_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ny*iy*i"), (char**)kwlist,
            &ctx_ptr, &sphere_buf, &sphere_count, &cyl_buf, &cyl_count))
        return NULL;

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    if (!check_rt_context(ctx) ||
        !check_geom_buffer(&sphere_buf, sphere_count, sizeof(float) * 4, "sphere") ||
        !check_geom_buffer(&cyl_buf, cyl_count, sizeof(float) * 8, "cylinder")) {
        PyBuffer_Release(&sphere_buf);
        PyBuffer_Release(&cyl_buf);
        return NULL;
    }
    id<MTLDevice> device = ctx->device;

    id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:sphere_buf.buf
                                          length:(NSUInteger)sphere_count * sizeof(float) * 4
                                         options:MTLResourceStorageModeShared];
    PyBuffer_Release(&sphere_buf);

    id<MTLBuffer> cylinderBuffer = nil;
    if (cyl_count > 0) {
        cylinderBuffer = [device newBufferWithBytes:cyl_buf.buf
                                     length:(NSUInteger)cyl_count * sizeof(float) * 8
                                    options:MTLResourceStorageModeShared];
    }
    PyBuffer_Release(&cyl_buf);
    if (!sphereBuffer || (cyl_count > 0 && !cylinderBuffer)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate geometry MTLBuffer");
        return NULL;
    }

    // Build AABBs for spheres.
    MTLAxisAlignedBoundingBox *sphere_aabbs = (MTLAxisAlignedBoundingBox *)malloc(
        sphere_count * sizeof(MTLAxisAlignedBoundingBox));
    float *sph = (float *)sphereBuffer.contents;
    for (int i = 0; i < sphere_count; i++) {
        float x = sph[i*4+0], y = sph[i*4+1], z = sph[i*4+2], r = sph[i*4+3];
        sphere_aabbs[i].min = MTLPackedFloat3Make(x-r, y-r, z-r);
        sphere_aabbs[i].max = MTLPackedFloat3Make(x+r, y+r, z+r);
    }
    id<MTLBuffer> sphereAABBBuf = [device newBufferWithBytes:sphere_aabbs
                                     length:sphere_count * sizeof(MTLAxisAlignedBoundingBox)
                                    options:MTLResourceStorageModeShared];
    free(sphere_aabbs);

    MTLAccelerationStructureBoundingBoxGeometryDescriptor *sphereGeom =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    sphereGeom.boundingBoxBuffer = sphereAABBBuf;
    sphereGeom.boundingBoxCount = sphere_count;
    sphereGeom.primitiveDataBuffer = sphereBuffer;
    sphereGeom.primitiveDataStride = sizeof(float) * 4;
    sphereGeom.primitiveDataElementSize = sizeof(float) * 4;
    sphereGeom.intersectionFunctionTableOffset = 0;

    NSMutableArray *geomDescs = [NSMutableArray arrayWithObject:sphereGeom];

    if (cyl_count > 0 && cylinderBuffer) {
        MTLAxisAlignedBoundingBox *cyl_aabbs = (MTLAxisAlignedBoundingBox *)malloc(
            cyl_count * sizeof(MTLAxisAlignedBoundingBox));
        float *cyl = (float *)cylinderBuffer.contents;
        for (int i = 0; i < cyl_count; i++) {
            float x0 = cyl[i*8+0], y0 = cyl[i*8+1], z0 = cyl[i*8+2], r = cyl[i*8+3];
            float x1 = cyl[i*8+4], y1 = cyl[i*8+5], z1 = cyl[i*8+6];
            float minx = fmin(x0, x1) - r, miny = fmin(y0, y1) - r, minz = fmin(z0, z1) - r;
            float maxx = fmax(x0, x1) + r, maxy = fmax(y0, y1) + r, maxz = fmax(z0, z1) + r;
            cyl_aabbs[i].min = MTLPackedFloat3Make(minx, miny, minz);
            cyl_aabbs[i].max = MTLPackedFloat3Make(maxx, maxy, maxz);
        }
        id<MTLBuffer> cylAABBBuf = [device newBufferWithBytes:cyl_aabbs
                                         length:cyl_count * sizeof(MTLAxisAlignedBoundingBox)
                                        options:MTLResourceStorageModeShared];
        free(cyl_aabbs);

        MTLAccelerationStructureBoundingBoxGeometryDescriptor *cylGeom =
            [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        cylGeom.boundingBoxBuffer = cylAABBBuf;
        cylGeom.boundingBoxCount = cyl_count;
        cylGeom.primitiveDataBuffer = cylinderBuffer;
        cylGeom.primitiveDataStride = sizeof(float) * 8;
        cylGeom.primitiveDataElementSize = sizeof(float) * 8;
        cylGeom.intersectionFunctionTableOffset = 1;

        [geomDescs addObject:cylGeom];
    }

    MTLPrimitiveAccelerationStructureDescriptor *asDesc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    asDesc.geometryDescriptors = geomDescs;

    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:asDesc];
    id<MTLAccelerationStructure> accelStruct =
        [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratchBuffer =
        [device newBufferWithLength:sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];

    id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> encoder =
        [cmdBuf accelerationStructureCommandEncoder];
    [encoder buildAccelerationStructure:accelStruct
                             descriptor:asDesc
                          scratchBuffer:scratchBuffer
                    scratchBufferOffset:0];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    long as_ptr = (long)(__bridge_retained void *)accelStruct;
    long sb_ptr = (long)(__bridge_retained void *)sphereBuffer;
    long cb_ptr = cylinderBuffer ? (long)(__bridge_retained void *)cylinderBuffer : 0;
    return Py_BuildValue("lll", as_ptr, sb_ptr, cb_ptr);
}

extern "C"
PyObject* metal_rt_build_scene_accel(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    Py_buffer sphere_buf, cyl_buf, tri_buf;
    int sphere_count, cyl_count, tri_count;
    static const char* kwlist[] = {"context",
                                   "sphere_data", "sphere_count",
                                   "cylinder_data", "cylinder_count",
                                   "triangle_data", "triangle_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ny*iy*iy*i"), (char**)kwlist,
            &ctx_ptr, &sphere_buf, &sphere_count,
            &cyl_buf, &cyl_count, &tri_buf, &tri_count))
        return NULL;

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    if (!check_rt_context(ctx) ||
        !check_geom_buffer(&sphere_buf, sphere_count, sizeof(float) * 4, "sphere") ||
        !check_geom_buffer(&cyl_buf, cyl_count, sizeof(float) * 8, "cylinder") ||
        !check_geom_buffer(&tri_buf, tri_count, sizeof(float) * 12, "triangle")) {
        PyBuffer_Release(&sphere_buf);
        PyBuffer_Release(&cyl_buf);
        PyBuffer_Release(&tri_buf);
        return NULL;
    }
    id<MTLDevice> device = ctx->device;

    id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:sphere_buf.buf
                                          length:(NSUInteger)sphere_count * sizeof(float) * 4
                                         options:MTLResourceStorageModeShared];
    PyBuffer_Release(&sphere_buf);

    id<MTLBuffer> cylinderBuffer = nil;
    if (cyl_count > 0) {
        cylinderBuffer = [device newBufferWithBytes:cyl_buf.buf
                                     length:(NSUInteger)cyl_count * sizeof(float) * 8
                                    options:MTLResourceStorageModeShared];
    }
    PyBuffer_Release(&cyl_buf);

    id<MTLBuffer> triangleBuffer = nil;
    if (tri_count > 0) {
        triangleBuffer = [device newBufferWithBytes:tri_buf.buf
                                     length:(NSUInteger)tri_count * sizeof(float) * 12
                                    options:MTLResourceStorageModeShared];
    }
    PyBuffer_Release(&tri_buf);

    if (!sphereBuffer || (cyl_count > 0 && !cylinderBuffer)
            || (tri_count > 0 && !triangleBuffer)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate geometry MTLBuffer");
        return NULL;
    }

    // Sphere AABBs.
    MTLAxisAlignedBoundingBox *sphere_aabbs = (MTLAxisAlignedBoundingBox *)malloc(
        sphere_count * sizeof(MTLAxisAlignedBoundingBox));
    float *sph = (float *)sphereBuffer.contents;
    for (int i = 0; i < sphere_count; i++) {
        float x = sph[i*4+0], y = sph[i*4+1], z = sph[i*4+2], r = sph[i*4+3];
        sphere_aabbs[i].min = MTLPackedFloat3Make(x-r, y-r, z-r);
        sphere_aabbs[i].max = MTLPackedFloat3Make(x+r, y+r, z+r);
    }
    id<MTLBuffer> sphereAABBBuf = [device newBufferWithBytes:sphere_aabbs
                                     length:sphere_count * sizeof(MTLAxisAlignedBoundingBox)
                                    options:MTLResourceStorageModeShared];
    free(sphere_aabbs);

    MTLAccelerationStructureBoundingBoxGeometryDescriptor *sphereGeom =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    sphereGeom.boundingBoxBuffer = sphereAABBBuf;
    sphereGeom.boundingBoxCount = sphere_count;
    sphereGeom.primitiveDataBuffer = sphereBuffer;
    sphereGeom.primitiveDataStride = sizeof(float) * 4;
    sphereGeom.primitiveDataElementSize = sizeof(float) * 4;
    sphereGeom.intersectionFunctionTableOffset = 0;

    NSMutableArray *geomDescs = [NSMutableArray arrayWithObject:sphereGeom];

    if (cyl_count > 0 && cylinderBuffer) {
        MTLAxisAlignedBoundingBox *cyl_aabbs = (MTLAxisAlignedBoundingBox *)malloc(
            cyl_count * sizeof(MTLAxisAlignedBoundingBox));
        float *cyl = (float *)cylinderBuffer.contents;
        for (int i = 0; i < cyl_count; i++) {
            float x0 = cyl[i*8+0], y0 = cyl[i*8+1], z0 = cyl[i*8+2], r = cyl[i*8+3];
            float x1 = cyl[i*8+4], y1 = cyl[i*8+5], z1 = cyl[i*8+6];
            float minx = fmin(x0, x1) - r, miny = fmin(y0, y1) - r, minz = fmin(z0, z1) - r;
            float maxx = fmax(x0, x1) + r, maxy = fmax(y0, y1) + r, maxz = fmax(z0, z1) + r;
            cyl_aabbs[i].min = MTLPackedFloat3Make(minx, miny, minz);
            cyl_aabbs[i].max = MTLPackedFloat3Make(maxx, maxy, maxz);
        }
        id<MTLBuffer> cylAABBBuf = [device newBufferWithBytes:cyl_aabbs
                                         length:cyl_count * sizeof(MTLAxisAlignedBoundingBox)
                                        options:MTLResourceStorageModeShared];
        free(cyl_aabbs);

        MTLAccelerationStructureBoundingBoxGeometryDescriptor *cylGeom =
            [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        cylGeom.boundingBoxBuffer = cylAABBBuf;
        cylGeom.boundingBoxCount = cyl_count;
        cylGeom.primitiveDataBuffer = cylinderBuffer;
        cylGeom.primitiveDataStride = sizeof(float) * 8;
        cylGeom.primitiveDataElementSize = sizeof(float) * 8;
        cylGeom.intersectionFunctionTableOffset = 1;
        [geomDescs addObject:cylGeom];
    } else {
        // Placeholder empty cylinder geometry so triangle gets geometry_id=2.
        // Metal requires at least 1 primitive per geometry, so add a
        // degenerate AABB if there are triangles but no cylinders.
        if (tri_count > 0) {
            MTLAxisAlignedBoundingBox dummy = {};
            dummy.min = MTLPackedFloat3Make(0, 0, 0);
            dummy.max = MTLPackedFloat3Make(0, 0, 0);
            id<MTLBuffer> dummyAABB = [device newBufferWithBytes:&dummy
                length:sizeof(MTLAxisAlignedBoundingBox)
                options:MTLResourceStorageModeShared];
            // Dummy 8-float primitive data (cylinder-shaped, zero radius).
            float dummyCyl[8] = {};
            id<MTLBuffer> dummyPrim = [device newBufferWithBytes:dummyCyl
                length:sizeof(dummyCyl)
                options:MTLResourceStorageModeShared];

            MTLAccelerationStructureBoundingBoxGeometryDescriptor *dummyGeom =
                [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
            dummyGeom.boundingBoxBuffer = dummyAABB;
            dummyGeom.boundingBoxCount = 1;
            dummyGeom.primitiveDataBuffer = dummyPrim;
            dummyGeom.primitiveDataStride = sizeof(float) * 8;
            dummyGeom.primitiveDataElementSize = sizeof(float) * 8;
            dummyGeom.intersectionFunctionTableOffset = 1;
            [geomDescs addObject:dummyGeom];
        }
    }

    if (tri_count > 0 && triangleBuffer) {
        MTLAxisAlignedBoundingBox *tri_aabbs = (MTLAxisAlignedBoundingBox *)malloc(
            tri_count * sizeof(MTLAxisAlignedBoundingBox));
        float *tri = (float *)triangleBuffer.contents;
        for (int i = 0; i < tri_count; i++) {
            float x0 = tri[i*12+0], y0 = tri[i*12+1], z0 = tri[i*12+2];
            float x1 = tri[i*12+4], y1 = tri[i*12+5], z1 = tri[i*12+6];
            float x2 = tri[i*12+8], y2 = tri[i*12+9], z2 = tri[i*12+10];
            float minx = fmin(fmin(x0, x1), x2) - 1e-4f;
            float miny = fmin(fmin(y0, y1), y2) - 1e-4f;
            float minz = fmin(fmin(z0, z1), z2) - 1e-4f;
            float maxx = fmax(fmax(x0, x1), x2) + 1e-4f;
            float maxy = fmax(fmax(y0, y1), y2) + 1e-4f;
            float maxz = fmax(fmax(z0, z1), z2) + 1e-4f;
            tri_aabbs[i].min = MTLPackedFloat3Make(minx, miny, minz);
            tri_aabbs[i].max = MTLPackedFloat3Make(maxx, maxy, maxz);
        }
        id<MTLBuffer> triAABBBuf = [device newBufferWithBytes:tri_aabbs
                                         length:tri_count * sizeof(MTLAxisAlignedBoundingBox)
                                        options:MTLResourceStorageModeShared];
        free(tri_aabbs);

        MTLAccelerationStructureBoundingBoxGeometryDescriptor *triGeom =
            [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        triGeom.boundingBoxBuffer = triAABBBuf;
        triGeom.boundingBoxCount = tri_count;
        triGeom.primitiveDataBuffer = triangleBuffer;
        triGeom.primitiveDataStride = sizeof(float) * 12;
        triGeom.primitiveDataElementSize = sizeof(float) * 12;
        triGeom.intersectionFunctionTableOffset = 2;
        [geomDescs addObject:triGeom];
    }

    MTLPrimitiveAccelerationStructureDescriptor *asDesc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    asDesc.geometryDescriptors = geomDescs;

    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:asDesc];
    id<MTLAccelerationStructure> accelStruct =
        [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratchBuffer =
        [device newBufferWithLength:sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];

    id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> encoder =
        [cmdBuf accelerationStructureCommandEncoder];
    [encoder buildAccelerationStructure:accelStruct
                             descriptor:asDesc
                          scratchBuffer:scratchBuffer
                    scratchBufferOffset:0];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    long as_ptr = (long)(__bridge_retained void *)accelStruct;
    long sb_ptr = (long)(__bridge_retained void *)sphereBuffer;
    long cb_ptr = cylinderBuffer ? (long)(__bridge_retained void *)cylinderBuffer : 0;
    long tb_ptr = triangleBuffer ? (long)(__bridge_retained void *)triangleBuffer : 0;
    return Py_BuildValue("llll", as_ptr, sb_ptr, cb_ptr, tb_ptr);
}

extern "C"
PyObject* metal_rt_destroy_accel_struct(PyObject* self, PyObject* args, PyObject* kwds) {
    long as_ptr, sb_ptr;
    long cb_ptr = 0, tb_ptr = 0;
    static const char* kwlist[] = {"accel_struct", "sphere_buffer",
                                   "cylinder_buffer", "triangle_buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ll|ll"), (char**)kwlist,
            &as_ptr, &sb_ptr, &cb_ptr, &tb_ptr))
        return NULL;

    if (as_ptr) {
        id<MTLAccelerationStructure> as =
            (__bridge_transfer id<MTLAccelerationStructure>)(void *)as_ptr;
        as = nil;
    }
    if (sb_ptr) {
        id<MTLBuffer> sb = (__bridge_transfer id<MTLBuffer>)(void *)sb_ptr;
        sb = nil;
    }
    // build_mixed_accel / build_scene_accel bridge-retain the cylinder and
    // triangle buffers too; without releasing them here every geometry
    // rebuild leaked both buffers.
    if (cb_ptr) {
        id<MTLBuffer> cb = (__bridge_transfer id<MTLBuffer>)(void *)cb_ptr;
        cb = nil;
    }
    if (tb_ptr) {
        id<MTLBuffer> tb = (__bridge_transfer id<MTLBuffer>)(void *)tb_ptr;
        tb = nil;
    }

    Py_RETURN_NONE;
}

// ===========================================================================
//  RTAO pipeline
// ===========================================================================

extern "C"
PyObject* metal_rt_compile_rtao(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    const char* shader_source;
    static const char* kwlist[] = {"context", "shader_source", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ns"), (char**)kwlist,
            &ctx_ptr, &shader_source))
        return NULL;

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    id<MTLDevice> device = ctx->device;

    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:shader_source];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        PyErr_Format(PyExc_RuntimeError, "RTAO shader compilation failed: %s",
                     [[error localizedDescription] UTF8String]);
        return NULL;
    }

    s_rtao_isect_func = [library newFunctionWithName:@"sphere_intersection"];
    if (!s_rtao_isect_func) {
        PyErr_SetString(PyExc_RuntimeError, "sphere_intersection function not found");
        return NULL;
    }

    MTLLinkedFunctions *linkedFuncs = [[MTLLinkedFunctions alloc] init];
    linkedFuncs.functions = @[s_rtao_isect_func];

    MTLComputePipelineDescriptor *pipeDesc = [[MTLComputePipelineDescriptor alloc] init];
    pipeDesc.computeFunction = [library newFunctionWithName:@"rtao_kernel"];
    pipeDesc.linkedFunctions = linkedFuncs;
    pipeDesc.maxCallStackDepth = 1;

    if (!pipeDesc.computeFunction) {
        PyErr_SetString(PyExc_RuntimeError, "rtao_kernel function not found");
        return NULL;
    }

    s_rtao_pipeline = [device newComputePipelineStateWithDescriptor:pipeDesc
                                                            options:0
                                                         reflection:nil
                                                              error:&error];
    if (!s_rtao_pipeline) {
        PyErr_Format(PyExc_RuntimeError, "RTAO compute pipeline failed: %s",
                     [[error localizedDescription] UTF8String]);
        return NULL;
    }

    MTLIntersectionFunctionTableDescriptor *iftDesc =
        [[MTLIntersectionFunctionTableDescriptor alloc] init];
    iftDesc.functionCount = 1;
    s_rtao_ift = [s_rtao_pipeline newIntersectionFunctionTableWithDescriptor:iftDesc];
    id<MTLFunctionHandle> isectHandle =
        [s_rtao_pipeline functionHandleWithFunction:s_rtao_isect_func];
    [s_rtao_ift setFunction:isectHandle atIndex:0];

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Dispatch RTAO.
//
//  Creates its own command buffer with timeline semaphore wait/signal for
//  GPU-GPU sync with Vulkan. flush_gpu must be called first so the depth
//  buffer is available and the timeline semaphore has been signalled.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_rt_dispatch_rtao(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    long as_ptr, sb_ptr, depth_tex_ptr, ao_tex_ptr, prev_tex_ptr, accum_tex_ptr;
    Py_buffer uniforms_buf;
    static const char* kwlist[] = {"context", "accel_struct", "sphere_buffer",
                                   "depth_texture", "ao_texture", "prev_texture",
                                   "accum_texture", "uniforms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nlllllly*"), (char**)kwlist,
            &ctx_ptr, &as_ptr, &sb_ptr, &depth_tex_ptr,
            &ao_tex_ptr, &prev_tex_ptr, &accum_tex_ptr, &uniforms_buf))
        return NULL;

    if (!s_rtao_pipeline) {
        PyBuffer_Release(&uniforms_buf);
        PyErr_SetString(PyExc_RuntimeError, "RTAO pipeline not compiled");
        return NULL;
    }

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    id<MTLDevice> device = ctx->device;

    id<MTLAccelerationStructure> accelStruct =
        (__bridge id<MTLAccelerationStructure>)(void *)as_ptr;
    id<MTLBuffer> sphereBuffer =
        (__bridge id<MTLBuffer>)(void *)sb_ptr;
    MetalRTTexture* depthTex = (MetalRTTexture*)depth_tex_ptr;
    MetalRTTexture* aoTex = (MetalRTTexture*)ao_tex_ptr;
    MetalRTTexture* prevTex = (MetalRTTexture*)prev_tex_ptr;
    MetalRTTexture* accumTex = (MetalRTTexture*)accum_tex_ptr;

    id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:uniforms_buf.buf
                                           length:uniforms_buf.len
                                          options:MTLResourceStorageModeShared];
    PyBuffer_Release(&uniforms_buf);

    id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];

    // GPU-side wait for Vulkan rasterization to finish.
    uint64_t wait_value = ctx->vk_ctx->rt_sync_value;
    if (ctx->syncEvent && wait_value > 0) {
        [cmdBuf encodeWaitForEvent:ctx->syncEvent value:wait_value];
    }

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    if (!enc) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create compute encoder");
        return NULL;
    }

    [enc setComputePipelineState:s_rtao_pipeline];
    [enc setTexture:depthTex->texture atIndex:0];
    [enc setTexture:aoTex->texture atIndex:1];
    [enc setTexture:prevTex->texture atIndex:2];
    [enc setTexture:accumTex->texture atIndex:3];
    [enc setAccelerationStructure:accelStruct atBufferIndex:0];
    [enc setBuffer:uniformBuffer offset:0 atIndex:1];
    [enc setBuffer:sphereBuffer offset:0 atIndex:2];
    [enc setIntersectionFunctionTable:s_rtao_ift atBufferIndex:3];

    [enc useResource:accelStruct usage:MTLResourceUsageRead];
    [enc useResource:sphereBuffer usage:MTLResourceUsageRead];
    [s_rtao_ift setBuffer:sphereBuffer offset:0 atIndex:0];

    uint w = aoTex->width;
    uint h = aoTex->height;
    MTLSize gridSize = MTLSizeMake(w, h, 1);
    NSUInteger threadW = s_rtao_pipeline.threadExecutionWidth;
    NSUInteger threadH = s_rtao_pipeline.maxTotalThreadsPerThreadgroup / threadW;
    MTLSize threadGroupSize = MTLSizeMake(threadW, threadH, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc endEncoding];

    // Signal completion so end_frame's Vulkan submit can wait on it.
    if (ctx->syncEvent) {
        uint64_t signal_value = wait_value + 1;
        [cmdBuf encodeSignalEvent:ctx->syncEvent value:signal_value];
        ctx->vk_ctx->rt_sync_value = signal_value;
        ctx->vk_ctx->rt_sync_pending = true;
    }

    [cmdBuf commit];

    Py_RETURN_NONE;
}

// ===========================================================================
//  Path tracer pipeline
// ===========================================================================

extern "C"
PyObject* metal_rt_compile_pathtracer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    const char* shader_source;
    static const char* kwlist[] = {"context", "shader_source", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ns"), (char**)kwlist,
            &ctx_ptr, &shader_source))
        return NULL;

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    id<MTLDevice> device = ctx->device;

    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:shader_source];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        PyErr_Format(PyExc_RuntimeError, "Path tracer shader compilation failed: %s",
                     [[error localizedDescription] UTF8String]);
        return NULL;
    }

    id<MTLFunction> sphereIsect = [library newFunctionWithName:@"sphere_intersection"];
    if (!sphereIsect) {
        PyErr_SetString(PyExc_RuntimeError, "sphere_intersection not found in pathtracer");
        return NULL;
    }
    id<MTLFunction> cylIsect = [library newFunctionWithName:@"cylinder_intersection"];
    if (!cylIsect) {
        PyErr_SetString(PyExc_RuntimeError, "cylinder_intersection not found in pathtracer");
        return NULL;
    }
    id<MTLFunction> triIsect = [library newFunctionWithName:@"triangle_intersection"];
    if (!triIsect) {
        PyErr_SetString(PyExc_RuntimeError, "triangle_intersection not found in pathtracer");
        return NULL;
    }

    MTLLinkedFunctions *linkedFuncs = [[MTLLinkedFunctions alloc] init];
    linkedFuncs.functions = @[sphereIsect, cylIsect, triIsect];

    MTLComputePipelineDescriptor *pipeDesc = [[MTLComputePipelineDescriptor alloc] init];
    pipeDesc.computeFunction = [library newFunctionWithName:@"pathtracer_kernel"];
    pipeDesc.linkedFunctions = linkedFuncs;
    pipeDesc.maxCallStackDepth = 2;

    if (!pipeDesc.computeFunction) {
        PyErr_SetString(PyExc_RuntimeError, "pathtracer_kernel not found");
        return NULL;
    }

    s_pt_pipeline = [device newComputePipelineStateWithDescriptor:pipeDesc
                                                          options:0
                                                       reflection:nil
                                                            error:&error];
    if (!s_pt_pipeline) {
        PyErr_Format(PyExc_RuntimeError, "Path tracer pipeline failed: %s",
                     [[error localizedDescription] UTF8String]);
        return NULL;
    }

    MTLIntersectionFunctionTableDescriptor *iftDesc =
        [[MTLIntersectionFunctionTableDescriptor alloc] init];
    iftDesc.functionCount = 3;
    s_pt_ift = [s_pt_pipeline newIntersectionFunctionTableWithDescriptor:iftDesc];
    [s_pt_ift setFunction:[s_pt_pipeline functionHandleWithFunction:sphereIsect]
                  atIndex:0];
    [s_pt_ift setFunction:[s_pt_pipeline functionHandleWithFunction:cylIsect]
                  atIndex:1];
    [s_pt_ift setFunction:[s_pt_pipeline functionHandleWithFunction:triIsect]
                  atIndex:2];

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_rt_dispatch_pathtracer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    long as_ptr, sb_ptr, color_tex_ptr, prev_tex_ptr, accum_tex_ptr;
    long color_buf_ptr, cyl_buf_ptr, cyl_color_buf_ptr;
    long tri_buf_ptr = 0, tri_color_buf_ptr = 0;
    long vol_density_tex_ptr = 0, vol_colormap_tex_ptr = 0;
    Py_ssize_t color_buf_offset = 0, cyl_color_buf_offset = 0, tri_color_buf_offset = 0;
    Py_buffer uniforms_buf;
    static const char* kwlist[] = {"context", "accel_struct", "sphere_buffer",
                                   "color_texture", "prev_texture", "accum_texture",
                                   "color_buffer", "color_buffer_offset",
                                   "cylinder_buffer",
                                   "cylinder_color_buffer", "cylinder_color_buffer_offset",
                                   "triangle_buffer",
                                   "triangle_color_buffer", "triangle_color_buffer_offset",
                                   "volume_density_texture", "volume_colormap_texture",
                                   "uniforms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nllllllnllnllnlly*"), (char**)kwlist,
            &ctx_ptr, &as_ptr, &sb_ptr, &color_tex_ptr, &prev_tex_ptr,
            &accum_tex_ptr, &color_buf_ptr, &color_buf_offset,
            &cyl_buf_ptr, &cyl_color_buf_ptr, &cyl_color_buf_offset,
            &tri_buf_ptr, &tri_color_buf_ptr, &tri_color_buf_offset,
            &vol_density_tex_ptr, &vol_colormap_tex_ptr,
            &uniforms_buf))
        return NULL;

    if (!s_pt_pipeline) {
        PyBuffer_Release(&uniforms_buf);
        PyErr_SetString(PyExc_RuntimeError, "Path tracer pipeline not compiled");
        return NULL;
    }

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    id<MTLDevice> device = ctx->device;

    id<MTLAccelerationStructure> accelStruct =
        (__bridge id<MTLAccelerationStructure>)(void *)as_ptr;
    id<MTLBuffer> sphereBuffer =
        (__bridge id<MTLBuffer>)(void *)sb_ptr;
    MetalRTTexture* colorTex = (MetalRTTexture*)color_tex_ptr;
    MetalRTTexture* prevTex = (MetalRTTexture*)prev_tex_ptr;
    MetalRTTexture* accumTex = (MetalRTTexture*)accum_tex_ptr;
    MetalRTTexture* volDensityTex = vol_density_tex_ptr
        ? (MetalRTTexture*)vol_density_tex_ptr : nil;
    MetalRTTexture* volColormapTex = vol_colormap_tex_ptr
        ? (MetalRTTexture*)vol_colormap_tex_ptr : nil;

    // Color buffer is an MTLBuffer* (bridged from Python long).
    id<MTLBuffer> colorBuffer = (__bridge id<MTLBuffer>)(void *)color_buf_ptr;
    id<MTLBuffer> cylinderBuffer = cyl_buf_ptr
        ? (__bridge id<MTLBuffer>)(void *)cyl_buf_ptr : nil;
    id<MTLBuffer> cylColorBuffer = cyl_color_buf_ptr
        ? (__bridge id<MTLBuffer>)(void *)cyl_color_buf_ptr : nil;
    id<MTLBuffer> triangleBuffer = tri_buf_ptr
        ? (__bridge id<MTLBuffer>)(void *)tri_buf_ptr : nil;
    id<MTLBuffer> triColorBuffer = tri_color_buf_ptr
        ? (__bridge id<MTLBuffer>)(void *)tri_color_buf_ptr : nil;

    id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:uniforms_buf.buf
                                           length:uniforms_buf.len
                                          options:MTLResourceStorageModeShared];
    PyBuffer_Release(&uniforms_buf);

    id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];

    // GPU-side wait for Vulkan work to finish.
    uint64_t wait_value = ctx->vk_ctx->rt_sync_value;
    if (ctx->syncEvent && wait_value > 0) {
        [cmdBuf encodeWaitForEvent:ctx->syncEvent value:wait_value];
    }

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    if (!enc) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create compute encoder");
        return NULL;
    }

    [enc setComputePipelineState:s_pt_pipeline];
    [enc setTexture:colorTex->texture atIndex:0];
    [enc setTexture:prevTex->texture atIndex:1];
    [enc setTexture:accumTex->texture atIndex:2];
    // Volume density (texture3d) + transfer-function colormap (texture1d).
    // The host always passes valid handles (1-texel dummies when no volume).
    if (volDensityTex)
        [enc setTexture:volDensityTex->texture atIndex:3];
    if (volColormapTex)
        [enc setTexture:volColormapTex->texture atIndex:4];

    [enc setAccelerationStructure:accelStruct atBufferIndex:0];
    [enc setBuffer:uniformBuffer offset:0 atIndex:1];
    [enc setBuffer:sphereBuffer offset:0 atIndex:2];
    [enc setIntersectionFunctionTable:s_pt_ift atBufferIndex:3];
    [enc setBuffer:colorBuffer offset:color_buf_offset atIndex:4];
    if (cylinderBuffer)
        [enc setBuffer:cylinderBuffer offset:0 atIndex:5];
    if (cylColorBuffer)
        [enc setBuffer:cylColorBuffer offset:cyl_color_buf_offset atIndex:6];
    if (triangleBuffer)
        [enc setBuffer:triangleBuffer offset:0 atIndex:7];
    if (triColorBuffer)
        [enc setBuffer:triColorBuffer offset:tri_color_buf_offset atIndex:8];

    [enc useResource:accelStruct usage:MTLResourceUsageRead];
    [enc useResource:sphereBuffer usage:MTLResourceUsageRead];
    if (cylinderBuffer)
        [enc useResource:cylinderBuffer usage:MTLResourceUsageRead];
    if (triangleBuffer)
        [enc useResource:triangleBuffer usage:MTLResourceUsageRead];
    [s_pt_ift setBuffer:sphereBuffer offset:0 atIndex:0];
    if (cylinderBuffer)
        [s_pt_ift setBuffer:cylinderBuffer offset:0 atIndex:1];
    if (triangleBuffer)
        [s_pt_ift setBuffer:triangleBuffer offset:0 atIndex:2];

    uint w = colorTex->width;
    uint h = colorTex->height;
    MTLSize gridSize = MTLSizeMake(w, h, 1);
    NSUInteger threadW = s_pt_pipeline.threadExecutionWidth;
    NSUInteger threadH = s_pt_pipeline.maxTotalThreadsPerThreadgroup / threadW;
    MTLSize threadGroupSize = MTLSizeMake(threadW, threadH, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc endEncoding];

    // Signal completion so end_frame's Vulkan submit can wait on it.
    if (ctx->syncEvent) {
        uint64_t signal_value = wait_value + 1;
        [cmdBuf encodeSignalEvent:ctx->syncEvent value:signal_value];
        ctx->vk_ctx->rt_sync_value = signal_value;
        ctx->vk_ctx->rt_sync_pending = true;
    }

    [cmdBuf commit];

    Py_RETURN_NONE;
}
