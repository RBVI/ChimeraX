/*
 * Metal ray tracing support: acceleration structure building,
 * RTAO pipeline compilation (once), and compute kernel dispatch.
 */

#include <Python.h>
#include "metal_internal.h"

// Cached RTAO pipeline state — compiled once, reused every frame.
static id<MTLComputePipelineState> s_rtao_pipeline = nil;
static id<MTLFunction> s_rtao_isect_func = nil;
static id<MTLIntersectionFunctionTable> s_rtao_ift = nil;

// ---------------------------------------------------------------------------
//  Build a primitive acceleration structure from sphere bounding boxes.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_build_sphere_accel(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    Py_buffer sphere_buf;
    int sphere_count;
    static const char* kwlist[] = {"context", "sphere_data", "sphere_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ly*i"), (char**)kwlist,
            &ctx_ptr, &sphere_buf, &sphere_count))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLDevice> device = ctx->device;

    if (![device supportsRaytracing]) {
        PyBuffer_Release(&sphere_buf);
        PyErr_SetString(PyExc_RuntimeError, "Metal device does not support ray tracing");
        return NULL;
    }

    id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:sphere_buf.buf
                                          length:sphere_count * sizeof(float) * 4
                                         options:MTLResourceStorageModeShared];
    PyBuffer_Release(&sphere_buf);

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
    geomDesc.intersectionFunctionTableOffset = 0;  // use entry 0 in the VFT

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

// ---------------------------------------------------------------------------
//  Build acceleration structure with spheres (geometry 0) and optionally
//  cylinders (geometry 1).  Returns (accel_struct, sphere_buffer,
//  cylinder_buffer) as longs.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_build_mixed_accel(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    Py_buffer sphere_buf, cyl_buf;
    int sphere_count, cyl_count;
    static const char* kwlist[] = {"context", "sphere_data", "sphere_count",
                                   "cylinder_data", "cylinder_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ly*iy*i"), (char**)kwlist,
            &ctx_ptr, &sphere_buf, &sphere_count, &cyl_buf, &cyl_count))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLDevice> device = ctx->device;

    if (![device supportsRaytracing]) {
        PyBuffer_Release(&sphere_buf);
        PyBuffer_Release(&cyl_buf);
        PyErr_SetString(PyExc_RuntimeError, "Device does not support ray tracing");
        return NULL;
    }

    // Sphere buffer: N x float4 (x, y, z, radius).
    id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:sphere_buf.buf
                                          length:sphere_count * sizeof(float) * 4
                                         options:MTLResourceStorageModeShared];
    PyBuffer_Release(&sphere_buf);

    // Cylinder buffer: N x 8 floats (p0.xyz, radius, p1.xyz, pad).
    id<MTLBuffer> cylinderBuffer = nil;
    if (cyl_count > 0) {
        cylinderBuffer = [device newBufferWithBytes:cyl_buf.buf
                                     length:cyl_count * sizeof(float) * 8
                                    options:MTLResourceStorageModeShared];
    }
    PyBuffer_Release(&cyl_buf);

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

    // Sphere geometry descriptor.
    MTLAccelerationStructureBoundingBoxGeometryDescriptor *sphereGeom =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    sphereGeom.boundingBoxBuffer = sphereAABBBuf;
    sphereGeom.boundingBoxCount = sphere_count;
    sphereGeom.primitiveDataBuffer = sphereBuffer;
    sphereGeom.primitiveDataStride = sizeof(float) * 4;
    sphereGeom.primitiveDataElementSize = sizeof(float) * 4;
    sphereGeom.intersectionFunctionTableOffset = 0;  // IFT entry 0 = sphere

    NSMutableArray *geomDescs = [NSMutableArray arrayWithObject:sphereGeom];

    // Cylinder geometry descriptor (optional).
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
        cylGeom.intersectionFunctionTableOffset = 1;  // IFT entry 1 = cylinder

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

// ---------------------------------------------------------------------------
//  Build acceleration structure with spheres (0), cylinders (1), triangles (2).
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_build_scene_accel(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    Py_buffer sphere_buf, cyl_buf, tri_buf;
    int sphere_count, cyl_count, tri_count;
    static const char* kwlist[] = {"context",
                                   "sphere_data", "sphere_count",
                                   "cylinder_data", "cylinder_count",
                                   "triangle_data", "triangle_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ly*iy*iy*i"), (char**)kwlist,
            &ctx_ptr, &sphere_buf, &sphere_count,
            &cyl_buf, &cyl_count, &tri_buf, &tri_count))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLDevice> device = ctx->device;

    if (![device supportsRaytracing]) {
        PyBuffer_Release(&sphere_buf);
        PyBuffer_Release(&cyl_buf);
        PyBuffer_Release(&tri_buf);
        PyErr_SetString(PyExc_RuntimeError, "Device does not support ray tracing");
        return NULL;
    }

    id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:sphere_buf.buf
                                          length:sphere_count * sizeof(float) * 4
                                         options:MTLResourceStorageModeShared];
    PyBuffer_Release(&sphere_buf);

    id<MTLBuffer> cylinderBuffer = nil;
    if (cyl_count > 0) {
        cylinderBuffer = [device newBufferWithBytes:cyl_buf.buf
                                     length:cyl_count * sizeof(float) * 8
                                    options:MTLResourceStorageModeShared];
    }
    PyBuffer_Release(&cyl_buf);

    id<MTLBuffer> triangleBuffer = nil;
    if (tri_count > 0) {
        triangleBuffer = [device newBufferWithBytes:tri_buf.buf
                                     length:tri_count * sizeof(float) * 12
                                    options:MTLResourceStorageModeShared];
    }
    PyBuffer_Release(&tri_buf);

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
            cyl_aabbs[i].min = MTLPackedFloat3Make(fmin(x0,x1)-r, fmin(y0,y1)-r, fmin(z0,z1)-r);
            cyl_aabbs[i].max = MTLPackedFloat3Make(fmax(x0,x1)+r, fmax(y0,y1)+r, fmax(z0,z1)+r);
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
    } else if (tri_count > 0) {
        // Placeholder empty cylinder geometry so triangles get geometry_id=2.
        MTLAxisAlignedBoundingBox dummy = {};
        dummy.min = MTLPackedFloat3Make(0, 0, 0);
        dummy.max = MTLPackedFloat3Make(0, 0, 0);
        id<MTLBuffer> dummyAABB = [device newBufferWithBytes:&dummy
            length:sizeof(MTLAxisAlignedBoundingBox) options:MTLResourceStorageModeShared];
        float dummyCyl[8] = {};
        id<MTLBuffer> dummyPrim = [device newBufferWithBytes:dummyCyl
            length:sizeof(dummyCyl) options:MTLResourceStorageModeShared];
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

    if (tri_count > 0 && triangleBuffer) {
        MTLAxisAlignedBoundingBox *tri_aabbs = (MTLAxisAlignedBoundingBox *)malloc(
            tri_count * sizeof(MTLAxisAlignedBoundingBox));
        float *tri = (float *)triangleBuffer.contents;
        for (int i = 0; i < tri_count; i++) {
            float x0 = tri[i*12+0], y0 = tri[i*12+1], z0 = tri[i*12+2];
            float x1 = tri[i*12+4], y1 = tri[i*12+5], z1 = tri[i*12+6];
            float x2 = tri[i*12+8], y2 = tri[i*12+9], z2 = tri[i*12+10];
            float pad = 1e-4f;
            tri_aabbs[i].min = MTLPackedFloat3Make(
                fmin(fmin(x0,x1),x2)-pad, fmin(fmin(y0,y1),y2)-pad, fmin(fmin(z0,z1),z2)-pad);
            tri_aabbs[i].max = MTLPackedFloat3Make(
                fmax(fmax(x0,x1),x2)+pad, fmax(fmax(y0,y1),y2)+pad, fmax(fmax(z0,z1),z2)+pad);
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

// ---------------------------------------------------------------------------
//  Compile the RTAO compute pipeline with linked intersection function.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_compile_rtao(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    const char* shader_source;
    static const char* kwlist[] = {"context", "shader_source", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ls"), (char**)kwlist,
            &ctx_ptr, &shader_source))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
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

    // Get intersection function for linking.
    s_rtao_isect_func = [library newFunctionWithName:@"sphere_intersection"];
    if (!s_rtao_isect_func) {
        PyErr_SetString(PyExc_RuntimeError, "sphere_intersection function not found");
        return NULL;
    }

    // Link intersection function into compute pipeline.
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

    // Create intersection function table so the intersector can call
    // sphere_intersection for bounding box geometry.
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
//  Dispatch the RTAO compute kernel (uses cached pipeline).
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_dispatch_rtao(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, as_ptr, sb_ptr, ao_tex_ptr, prev_tex_ptr, accum_tex_ptr;
    Py_buffer uniforms_buf;
    static const char* kwlist[] = {"context", "accel_struct", "sphere_buffer",
                                   "ao_texture", "prev_texture", "accum_texture",
                                   "uniforms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lllllly*"), (char**)kwlist,
            &ctx_ptr, &as_ptr, &sb_ptr, &ao_tex_ptr, &prev_tex_ptr,
            &accum_tex_ptr, &uniforms_buf))
        return NULL;

    if (!s_rtao_pipeline) {
        PyBuffer_Release(&uniforms_buf);
        PyErr_SetString(PyExc_RuntimeError, "RTAO pipeline not compiled");
        return NULL;
    }

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLDevice> device = ctx->device;

    id<MTLAccelerationStructure> accelStruct =
        (__bridge id<MTLAccelerationStructure>)(void *)as_ptr;
    id<MTLBuffer> sphereBuffer =
        (__bridge id<MTLBuffer>)(void *)sb_ptr;
    MetalTexture* aoTex = (MetalTexture*)ao_tex_ptr;
    MetalTexture* prevTex = (MetalTexture*)prev_tex_ptr;
    MetalTexture* accumTex = (MetalTexture*)accum_tex_ptr;

    id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:uniforms_buf.buf
                                           length:uniforms_buf.len
                                          options:MTLResourceStorageModeShared];
    PyBuffer_Release(&uniforms_buf);

    // End any active render encoder.
    if (ctx->currentEncoder) {
        [ctx->currentEncoder endEncoding];
        ctx->currentEncoder = nil;
    }

    id<MTLComputeCommandEncoder> computeEncoder =
        [ctx->currentCommandBuffer computeCommandEncoder];
    if (!computeEncoder) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create compute encoder");
        return NULL;
    }

    [computeEncoder setComputePipelineState:s_rtao_pipeline];
    [computeEncoder setTexture:ctx->depthTexture atIndex:0];
    [computeEncoder setTexture:aoTex->texture atIndex:1];
    [computeEncoder setTexture:prevTex->texture atIndex:2];
    [computeEncoder setTexture:accumTex->texture atIndex:3];
    [computeEncoder setAccelerationStructure:accelStruct atBufferIndex:0];
    [computeEncoder setBuffer:uniformBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:sphereBuffer offset:0 atIndex:2];

    // Bind intersection function table for bounding box geometry.
    [computeEncoder setIntersectionFunctionTable:s_rtao_ift atBufferIndex:3];

    // Make resources visible to the intersection function.
    [computeEncoder useResource:accelStruct usage:MTLResourceUsageRead];
    [computeEncoder useResource:sphereBuffer usage:MTLResourceUsageRead];

    // The intersection function reads primitive data from the AS, but
    // the sphere buffer must also be explicitly marked as used.
    [s_rtao_ift setBuffer:sphereBuffer offset:0 atIndex:0];

    uint w = aoTex->width;
    uint h = aoTex->height;
    MTLSize gridSize = MTLSizeMake(w, h, 1);
    NSUInteger threadW = s_rtao_pipeline.threadExecutionWidth;
    NSUInteger threadH = s_rtao_pipeline.maxTotalThreadsPerThreadgroup / threadW;
    MTLSize threadGroupSize = MTLSizeMake(threadW, threadH, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Path tracer pipeline (compiled separately from RTAO).
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> s_pt_pipeline = nil;
static id<MTLFunction> s_pt_isect_func = nil;
static id<MTLIntersectionFunctionTable> s_pt_ift = nil;

extern "C"
PyObject* metal_compile_pathtracer(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    const char* shader_source;
    static const char* kwlist[] = {"context", "shader_source", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ls"), (char**)kwlist,
            &ctx_ptr, &shader_source))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
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

    // IFT with 3 entries: index 0 = spheres, 1 = cylinders, 2 = triangles.
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

// ---------------------------------------------------------------------------
//  Dispatch the path tracer compute kernel.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_dispatch_pathtracer(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, as_ptr, sb_ptr, color_tex_ptr, prev_tex_ptr, accum_tex_ptr;
    long color_buf_ptr, cyl_buf_ptr, cyl_color_buf_ptr;
    long tri_buf_ptr = 0, tri_color_buf_ptr = 0;
    Py_buffer uniforms_buf;
    static const char* kwlist[] = {"context", "accel_struct", "sphere_buffer",
                                   "color_texture", "prev_texture", "accum_texture",
                                   "color_buffer", "cylinder_buffer",
                                   "cylinder_color_buffer",
                                   "triangle_buffer", "triangle_color_buffer",
                                   "uniforms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llllllllllly*"), (char**)kwlist,
            &ctx_ptr, &as_ptr, &sb_ptr, &color_tex_ptr, &prev_tex_ptr,
            &accum_tex_ptr, &color_buf_ptr, &cyl_buf_ptr, &cyl_color_buf_ptr,
            &tri_buf_ptr, &tri_color_buf_ptr,
            &uniforms_buf))
        return NULL;

    if (!s_pt_pipeline) {
        PyBuffer_Release(&uniforms_buf);
        PyErr_SetString(PyExc_RuntimeError, "Path tracer pipeline not compiled");
        return NULL;
    }

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLDevice> device = ctx->device;

    id<MTLAccelerationStructure> accelStruct =
        (__bridge id<MTLAccelerationStructure>)(void *)as_ptr;
    id<MTLBuffer> sphereBuffer =
        (__bridge id<MTLBuffer>)(void *)sb_ptr;
    MetalTexture* colorTex = (MetalTexture*)color_tex_ptr;
    MetalTexture* prevTex = (MetalTexture*)prev_tex_ptr;
    MetalTexture* accumTex = (MetalTexture*)accum_tex_ptr;
    MetalBuffer* colorBufWrapper = (MetalBuffer*)color_buf_ptr;
    id<MTLBuffer> colorBuffer = colorBufWrapper->buffer;

    id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:uniforms_buf.buf
                                           length:uniforms_buf.len
                                          options:MTLResourceStorageModeShared];
    PyBuffer_Release(&uniforms_buf);

    if (ctx->currentEncoder) {
        [ctx->currentEncoder endEncoding];
        ctx->currentEncoder = nil;
    }

    id<MTLComputeCommandEncoder> enc =
        [ctx->currentCommandBuffer computeCommandEncoder];
    if (!enc) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create compute encoder");
        return NULL;
    }

    [enc setComputePipelineState:s_pt_pipeline];
    [enc setTexture:colorTex->texture atIndex:0];
    [enc setTexture:prevTex->texture atIndex:1];
    [enc setTexture:accumTex->texture atIndex:2];
    id<MTLBuffer> cylinderBuffer = cyl_buf_ptr
        ? (__bridge id<MTLBuffer>)(void *)cyl_buf_ptr : nil;
    MetalBuffer* cylColorWrapper = cyl_color_buf_ptr
        ? (MetalBuffer*)cyl_color_buf_ptr : nil;
    id<MTLBuffer> triangleBuffer = tri_buf_ptr
        ? (__bridge id<MTLBuffer>)(void *)tri_buf_ptr : nil;
    MetalBuffer* triColorWrapper = tri_color_buf_ptr
        ? (MetalBuffer*)tri_color_buf_ptr : nil;

    [enc setAccelerationStructure:accelStruct atBufferIndex:0];
    [enc setBuffer:uniformBuffer offset:0 atIndex:1];
    [enc setBuffer:sphereBuffer offset:0 atIndex:2];
    [enc setIntersectionFunctionTable:s_pt_ift atBufferIndex:3];
    [enc setBuffer:colorBuffer offset:0 atIndex:4];
    if (cylinderBuffer)
        [enc setBuffer:cylinderBuffer offset:0 atIndex:5];
    if (cylColorWrapper)
        [enc setBuffer:cylColorWrapper->buffer offset:0 atIndex:6];
    if (triangleBuffer)
        [enc setBuffer:triangleBuffer offset:0 atIndex:7];
    if (triColorWrapper)
        [enc setBuffer:triColorWrapper->buffer offset:0 atIndex:8];

    [enc useResource:accelStruct usage:MTLResourceUsageRead];
    [enc useResource:sphereBuffer usage:MTLResourceUsageRead];
    if (cylinderBuffer)
        [enc useResource:cylinderBuffer usage:MTLResourceUsageRead];
    if (triangleBuffer)
        [enc useResource:triangleBuffer usage:MTLResourceUsageRead];
    // Set sphere data on IFT entry 0 (sphere intersection reads it).
    [s_pt_ift setBuffer:sphereBuffer offset:0 atIndex:0];
    // Set cylinder data on IFT entry 1 (cylinder intersection reads it).
    if (cylinderBuffer)
        [s_pt_ift setBuffer:cylinderBuffer offset:0 atIndex:1];
    // Set triangle data on IFT entry 2 (triangle intersection reads it).
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

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Destroy acceleration structure and sphere buffer.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_destroy_accel_struct(PyObject* self, PyObject* args, PyObject* kwds) {
    long as_ptr, sb_ptr;
    static const char* kwlist[] = {"accel_struct", "sphere_buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ll"), (char**)kwlist,
            &as_ptr, &sb_ptr))
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

    Py_RETURN_NONE;
}
