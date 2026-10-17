/*
 * Vulkan ray tracing support: acceleration structure building, RTAO and
 * path-tracer compute pipeline compilation, and per-frame dispatch.
 *
 * Mirrors the Metal raytracing module (metal_raytracing.m). Uses Vulkan
 * ray queries (VK_KHR_ray_query) from a compute shader rather than a
 * full ray-tracing pipeline — a single .comp shader walks AABB candidates
 * and runs analytical sphere/cylinder intersection inline, which avoids
 * shader binding tables and per-primitive intersection shaders entirely.
 */

#include <Python.h>
#include "vulkan_internal.h"
#include "vulkan_api.h"

#include <shaderc/shaderc.h>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
//  Owned state for a built acceleration structure. Ray queries require a
//  top-level AS, so we build a 1-instance TLAS that wraps a single BLAS
//  containing all sphere (and optionally cylinder) geometries. Returned to
//  Python as a Py_ssize_t pointer; freed via vulkan_destroy_accel_struct.
// ---------------------------------------------------------------------------
struct VkRtAccelStructure {
    VulkanContext* ctx;
    // Bottom-level AS (up to three AABB geometries: spheres, cylinders,
    // triangles).
    VkAccelerationStructureKHR blas;
    VkBuffer            blas_buffer;
    VmaAllocation       blas_buffer_alloc;
    // Top-level AS holding a single identity-transform instance of `blas`.
    // This is what gets bound to rayQueryEXT in the compute shader.
    VkAccelerationStructureKHR tlas;
    VkBuffer            tlas_buffer;
    VmaAllocation       tlas_buffer_alloc;
    VkBuffer            instance_buffer;
    VmaAllocation       instance_buffer_alloc;
    // Per-primitive data (referenced by the compute shader for analytical
    // intersection and per-primitive colors).
    VkBuffer            sphere_buffer;
    VmaAllocation       sphere_buffer_alloc;
    uint32_t            sphere_count;
    VkBuffer            cylinder_buffer;
    VmaAllocation       cylinder_buffer_alloc;
    uint32_t            cylinder_count;
    VkBuffer            triangle_buffer;
    VmaAllocation       triangle_buffer_alloc;
    uint32_t            triangle_count;
};

// Short aliases keep the pipeline code readable while the actual state is
// owned by the current VulkanContext (and therefore by its VkDevice).
#define s_rtao          (ctx->rtao_pipeline)
#define s_rtao_compiled (ctx->rtao_pipeline.compiled)
#define s_pt            (ctx->pathtracer_pipeline)
#define s_pt_compiled   (ctx->pathtracer_pipeline.compiled)

static void destroy_rtao_pipeline_state(VulkanContext* ctx)
{
    VkRtaoPipelineState& state = ctx->rtao_pipeline;
    if (state.pipeline) vkDestroyPipeline(ctx->device, state.pipeline, nullptr);
    if (state.sampler) vkDestroySampler(ctx->device, state.sampler, nullptr);
    if (state.descriptor_pool)
        vkDestroyDescriptorPool(ctx->device, state.descriptor_pool, nullptr);
    if (state.pipeline_layout)
        vkDestroyPipelineLayout(ctx->device, state.pipeline_layout, nullptr);
    if (state.set_layout)
        vkDestroyDescriptorSetLayout(ctx->device, state.set_layout, nullptr);
    state = {};
}

static void destroy_pt_pipeline_state(VulkanContext* ctx)
{
    VkPathTracerPipelineState& state = ctx->pathtracer_pipeline;
    if (state.pipeline) vkDestroyPipeline(ctx->device, state.pipeline, nullptr);
    if (state.sampler) vkDestroySampler(ctx->device, state.sampler, nullptr);
    if (state.descriptor_pool)
        vkDestroyDescriptorPool(ctx->device, state.descriptor_pool, nullptr);
    if (state.pipeline_layout)
        vkDestroyPipelineLayout(ctx->device, state.pipeline_layout, nullptr);
    if (state.set_layout)
        vkDestroyDescriptorSetLayout(ctx->device, state.set_layout, nullptr);
    if (state.dummy_buffer)
        vmaDestroyBuffer(ctx->allocator, state.dummy_buffer,
                         state.dummy_buffer_alloc);
    state = {};
}

// ---------------------------------------------------------------------------
//  Small helpers
// ---------------------------------------------------------------------------

// Allocate a GPU-local buffer with the requested usage. Used for AS storage
// and scratch buffers. Returns false on failure.
static bool create_gpu_buffer(VulkanContext* ctx, VkDeviceSize size,
                              VkBufferUsageFlags usage,
                              VkBuffer* out_buffer,
                              VmaAllocation* out_allocation)
{
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    return vmaCreateBuffer(ctx->allocator, &bi, &ai,
                           out_buffer, out_allocation, nullptr) == VK_SUCCESS;
}

// Allocate a host-visible buffer pre-filled with src data. Used for the
// AABB and primitive-data buffers handed to the AS builder, and for the
// per-frame uniform buffer of the compute kernel.
static bool create_host_buffer(VulkanContext* ctx, const void* src,
                               VkDeviceSize size, VkBufferUsageFlags usage,
                               VkBuffer* out_buffer,
                               VmaAllocation* out_allocation)
{
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
             | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(ctx->allocator, &bi, &ai,
                        out_buffer, out_allocation, &info) != VK_SUCCESS)
        return false;
    if (src && info.pMappedData)
        memcpy(info.pMappedData, src, size);
    else if (src) {
        void* mapped = nullptr;
        vmaMapMemory(ctx->allocator, *out_allocation, &mapped);
        memcpy(mapped, src, size);
        vmaUnmapMemory(ctx->allocator, *out_allocation);
    }
    // HOST_ACCESS_SEQUENTIAL_WRITE may pick a non-coherent memory type;
    // make the writes visible to the GPU. No-op on coherent memory.
    if (src)
        vmaFlushAllocation(ctx->allocator, *out_allocation, 0, VK_WHOLE_SIZE);
    return true;
}

// Scratch buffers for acceleration-structure builds. The build reads the
// scratch via its device address, which the spec requires to be aligned to
// minAccelerationStructureScratchOffsetAlignment — plain vmaCreateBuffer
// only honors VkMemoryRequirements::alignment, which may be smaller.
static bool create_scratch_buffer(VulkanContext* ctx, VkDeviceSize size,
                                  VkBuffer* out_buffer,
                                  VmaAllocation* out_allocation)
{
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkDeviceSize alignment = ctx->min_as_scratch_alignment
        ? ctx->min_as_scratch_alignment : 1;
    return vmaCreateBufferWithAlignment(ctx->allocator, &bi, &ai, alignment,
                                        out_buffer, out_allocation,
                                        nullptr) == VK_SUCCESS;
}

// Look up a buffer's GPU address (required for AS scratch/data references).
static VkDeviceAddress buffer_device_address(VulkanContext* ctx, VkBuffer b) {
    VkBufferDeviceAddressInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bi.buffer = b;
    return vkGetBufferDeviceAddress(ctx->device, &bi);
}

// One-shot command buffer for AS builds, image transitions, etc. Mirrors
// the helpers in vulkan_texture.cpp but lives here to keep raytracing
// self-contained.
static VkCommandBuffer rt_begin_one_shot(VulkanContext* ctx) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = ctx->command_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx->device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

static void rt_end_one_shot(VulkanContext* ctx, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx->graphics_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->graphics_queue);
    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
}

// Compute AABBs for a flat array of (x, y, z, radius) spheres.
// Pad AABBs by a small fraction of the primitive radius. Vulkan BVH
// builders are free to use compressed/quantised AABB representations,
// which can shrink the actual bounding region slightly. With tight
// AABBs that quantisation loss can leave silhouette-edge pixels just
// outside the BVH leaf, so the analytical sphere is never visited and
// the atom appears "erased" in those pixels — exactly the artefact
// observed when Metal's intersector (no quantisation) renders the
// same scene cleanly. The padding here is small enough not to affect
// rendering of pixels that legitimately miss the analytical primitive.
static constexpr float AABB_PAD_FRAC = 1e-3f;

static void fill_sphere_aabbs(const float* sph, uint32_t count,
                              VkAabbPositionsKHR* out) {
    for (uint32_t i = 0; i < count; i++) {
        float x = sph[i*4+0], y = sph[i*4+1], z = sph[i*4+2], r = sph[i*4+3];
        float pad = r * AABB_PAD_FRAC + 1e-4f;
        out[i].minX = x - r - pad; out[i].minY = y - r - pad; out[i].minZ = z - r - pad;
        out[i].maxX = x + r + pad; out[i].maxY = y + r + pad; out[i].maxZ = z + r + pad;
    }
}

// Compute AABBs for triangle primitives packed as 12 floats each:
// (v0.x, v0.y, v0.z, 0, v1.x, v1.y, v1.z, 0, v2.x, v2.y, v2.z, 0).
static void fill_triangle_aabbs(const float* tri, uint32_t count,
                                VkAabbPositionsKHR* out) {
    for (uint32_t i = 0; i < count; i++) {
        float x0 = tri[i*12+0], y0 = tri[i*12+1], z0 = tri[i*12+2];
        float x1 = tri[i*12+4], y1 = tri[i*12+5], z1 = tri[i*12+6];
        float x2 = tri[i*12+8], y2 = tri[i*12+9], z2 = tri[i*12+10];
        float pad = 1e-4f;
        float minx = fminf(fminf(x0, x1), x2) - pad;
        float miny = fminf(fminf(y0, y1), y2) - pad;
        float minz = fminf(fminf(z0, z1), z2) - pad;
        float maxx = fmaxf(fmaxf(x0, x1), x2) + pad;
        float maxy = fmaxf(fmaxf(y0, y1), y2) + pad;
        float maxz = fmaxf(fmaxf(z0, z1), z2) + pad;
        out[i].minX = minx; out[i].minY = miny; out[i].minZ = minz;
        out[i].maxX = maxx; out[i].maxY = maxy; out[i].maxZ = maxz;
    }
}

// Compute AABBs for a flat array of cylinders packed as
// (p0.xyz, radius, p1.xyz, pad) — same layout as the Metal backend.
static void fill_cylinder_aabbs(const float* cyl, uint32_t count,
                                VkAabbPositionsKHR* out) {
    for (uint32_t i = 0; i < count; i++) {
        float x0 = cyl[i*8+0], y0 = cyl[i*8+1], z0 = cyl[i*8+2], r = cyl[i*8+3];
        float x1 = cyl[i*8+4], y1 = cyl[i*8+5], z1 = cyl[i*8+6];
        float pad = r * AABB_PAD_FRAC + 1e-4f;
        float minx = (x0 < x1 ? x0 : x1) - r - pad;
        float miny = (y0 < y1 ? y0 : y1) - r - pad;
        float minz = (z0 < z1 ? z0 : z1) - r - pad;
        float maxx = (x0 > x1 ? x0 : x1) + r + pad;
        float maxy = (y0 > y1 ? y0 : y1) + r + pad;
        float maxz = (z0 > z1 ? z0 : z1) + r + pad;
        out[i].minX = minx; out[i].minY = miny; out[i].minZ = minz;
        out[i].maxX = maxx; out[i].maxY = maxy; out[i].maxZ = maxz;
    }
}

// Build a bottom-level acceleration structure with one or two AABB
// geometries; sets out->blas + out->blas_buffer.
static bool build_blas(VulkanContext* ctx,
                       VkBuffer aabb_buffers[3],
                       uint32_t prim_counts[3],
                       uint32_t geometry_count,
                       VkRtAccelStructure* out)
{
    VkAccelerationStructureGeometryKHR geometries[3] = {};
    VkAccelerationStructureBuildRangeInfoKHR ranges[3] = {};
    const VkAccelerationStructureBuildRangeInfoKHR* range_ptrs[1];

    for (uint32_t g = 0; g < geometry_count; g++) {
        geometries[g].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometries[g].geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        geometries[g].flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometries[g].geometry.aabbs.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        geometries[g].geometry.aabbs.data.deviceAddress =
            buffer_device_address(ctx, aabb_buffers[g]);
        geometries[g].geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

        ranges[g].primitiveCount = prim_counts[g];
    }
    range_ptrs[0] = ranges;

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = geometry_count;
    build_info.pGeometries = geometries;

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    ctx->pfn_get_as_build_sizes(ctx->device,
                                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                &build_info, prim_counts, &sizes);

    if (!create_gpu_buffer(ctx, sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            &out->blas_buffer, &out->blas_buffer_alloc))
        return false;

    VkAccelerationStructureCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    ci.buffer = out->blas_buffer;
    ci.size = sizes.accelerationStructureSize;
    ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (ctx->pfn_create_as(ctx->device, &ci, nullptr, &out->blas) != VK_SUCCESS) {
        vmaDestroyBuffer(ctx->allocator, out->blas_buffer, out->blas_buffer_alloc);
        return false;
    }

    VkBuffer scratch = VK_NULL_HANDLE;
    VmaAllocation scratch_alloc = VK_NULL_HANDLE;
    if (!create_scratch_buffer(ctx, sizes.buildScratchSize,
                               &scratch, &scratch_alloc)) {
        ctx->pfn_destroy_as(ctx->device, out->blas, nullptr);
        vmaDestroyBuffer(ctx->allocator, out->blas_buffer, out->blas_buffer_alloc);
        return false;
    }
    build_info.dstAccelerationStructure = out->blas;
    build_info.scratchData.deviceAddress = buffer_device_address(ctx, scratch);

    VkCommandBuffer cmd = rt_begin_one_shot(ctx);
    ctx->pfn_cmd_build_as(cmd, 1, &build_info, range_ptrs);
    rt_end_one_shot(ctx, cmd);

    vmaDestroyBuffer(ctx->allocator, scratch, scratch_alloc);
    return true;
}

// Error-path cleanup for build_tlas: the instance buffer is created first,
// so every later failure must release it — callers only clean up the BLAS.
static void destroy_tlas_instance_buffer(VulkanContext* ctx, VkRtAccelStructure* out)
{
    if (out->instance_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx->allocator, out->instance_buffer,
                         out->instance_buffer_alloc);
        out->instance_buffer = VK_NULL_HANDLE;
        out->instance_buffer_alloc = VK_NULL_HANDLE;
    }
}

// Build a 1-instance top-level AS that references out->blas with an
// identity transform. Sets out->tlas, out->tlas_buffer, out->instance_buffer.
static bool build_tlas(VulkanContext* ctx, VkRtAccelStructure* out)
{
    VkAccelerationStructureDeviceAddressInfoKHR ai{};
    ai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    ai.accelerationStructure = out->blas;
    VkDeviceAddress blas_addr = ctx->pfn_get_as_device_address(ctx->device, &ai);

    VkAccelerationStructureInstanceKHR inst{};
    // Identity 3x4 transform (row-major).
    inst.transform.matrix[0][0] = 1.0f;
    inst.transform.matrix[1][1] = 1.0f;
    inst.transform.matrix[2][2] = 1.0f;
    inst.instanceCustomIndex = 0;
    inst.mask = 0xFF;
    inst.instanceShaderBindingTableRecordOffset = 0;
    inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst.accelerationStructureReference = blas_addr;

    if (!create_host_buffer(ctx, &inst, sizeof(inst),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            &out->instance_buffer, &out->instance_buffer_alloc))
        return false;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.arrayOfPointers = VK_FALSE;
    geom.geometry.instances.data.deviceAddress =
        buffer_device_address(ctx, out->instance_buffer);

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries = &geom;

    uint32_t inst_count = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    ctx->pfn_get_as_build_sizes(ctx->device,
                                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                &build_info, &inst_count, &sizes);

    if (!create_gpu_buffer(ctx, sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            &out->tlas_buffer, &out->tlas_buffer_alloc)) {
        destroy_tlas_instance_buffer(ctx, out);
        return false;
    }

    VkAccelerationStructureCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    ci.buffer = out->tlas_buffer;
    ci.size = sizes.accelerationStructureSize;
    ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (ctx->pfn_create_as(ctx->device, &ci, nullptr, &out->tlas) != VK_SUCCESS) {
        vmaDestroyBuffer(ctx->allocator, out->tlas_buffer, out->tlas_buffer_alloc);
        destroy_tlas_instance_buffer(ctx, out);
        return false;
    }

    VkBuffer scratch = VK_NULL_HANDLE;
    VmaAllocation scratch_alloc = VK_NULL_HANDLE;
    if (!create_scratch_buffer(ctx, sizes.buildScratchSize,
                               &scratch, &scratch_alloc)) {
        ctx->pfn_destroy_as(ctx->device, out->tlas, nullptr);
        vmaDestroyBuffer(ctx->allocator, out->tlas_buffer, out->tlas_buffer_alloc);
        destroy_tlas_instance_buffer(ctx, out);
        return false;
    }
    build_info.dstAccelerationStructure = out->tlas;
    build_info.scratchData.deviceAddress = buffer_device_address(ctx, scratch);

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = 1;
    const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = &range;

    VkCommandBuffer cmd = rt_begin_one_shot(ctx);
    ctx->pfn_cmd_build_as(cmd, 1, &build_info, &range_ptr);
    rt_end_one_shot(ctx, cmd);

    vmaDestroyBuffer(ctx->allocator, scratch, scratch_alloc);
    return true;
}

// Convenience: build BLAS + wrapping TLAS.
static bool build_primitive_as(VulkanContext* ctx,
                               VkBuffer aabb_buffers[3],
                               uint32_t prim_counts[3],
                               uint32_t geometry_count,
                               VkRtAccelStructure* out)
{
    if (!build_blas(ctx, aabb_buffers, prim_counts, geometry_count, out))
        return false;
    if (!build_tlas(ctx, out)) {
        ctx->pfn_destroy_as(ctx->device, out->blas, nullptr);
        vmaDestroyBuffer(ctx->allocator, out->blas_buffer, out->blas_buffer_alloc);
        out->blas = VK_NULL_HANDLE;
        out->blas_buffer = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
//  build_sphere_accel — sphere-only BVH.
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_build_sphere_accel(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    Py_buffer sphere_buf;
    int sphere_count;
    static const char* kwlist[] = {"context", "sphere_data", "sphere_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ny*i"), (char**)kwlist,
            &ctx_ptr, &sphere_buf, &sphere_count))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("RT build_sphere_accel sphere_count=%d", sphere_count);
    if (!ctx->raytracing_supported) {
        PyBuffer_Release(&sphere_buf);
        PyErr_SetString(PyExc_RuntimeError, "Vulkan device does not support ray tracing");
        return NULL;
    }

    // Per-primitive sphere data (also bound to the compute shader as an SSBO).
    VkBuffer sphere_data_buf = VK_NULL_HANDLE;
    VmaAllocation sphere_data_alloc = VK_NULL_HANDLE;
    if (!create_host_buffer(ctx, sphere_buf.buf,
            sphere_count * sizeof(float) * 4,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            &sphere_data_buf, &sphere_data_alloc)) {
        PyBuffer_Release(&sphere_buf);
        PyErr_SetString(PyExc_RuntimeError, "Sphere data buffer allocation failed");
        return NULL;
    }

    // AABB buffer — fed to the AS builder, throwaway after the build.
    std::vector<VkAabbPositionsKHR> aabbs(sphere_count);
    fill_sphere_aabbs(reinterpret_cast<const float*>(sphere_buf.buf),
                      sphere_count, aabbs.data());
    PyBuffer_Release(&sphere_buf);

    VkBuffer aabb_buf = VK_NULL_HANDLE;
    VmaAllocation aabb_alloc = VK_NULL_HANDLE;
    if (!create_host_buffer(ctx, aabbs.data(),
            aabbs.size() * sizeof(VkAabbPositionsKHR),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            &aabb_buf, &aabb_alloc)) {
        vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
        PyErr_SetString(PyExc_RuntimeError, "AABB buffer allocation failed");
        return NULL;
    }

    VkRtAccelStructure* rt = new VkRtAccelStructure();
    rt->ctx = ctx;
    rt->sphere_buffer = sphere_data_buf;
    rt->sphere_buffer_alloc = sphere_data_alloc;
    rt->sphere_count = static_cast<uint32_t>(sphere_count);
    rt->cylinder_buffer = VK_NULL_HANDLE;
    rt->cylinder_buffer_alloc = VK_NULL_HANDLE;
    rt->cylinder_count = 0;
    rt->triangle_buffer = VK_NULL_HANDLE;
    rt->triangle_buffer_alloc = VK_NULL_HANDLE;
    rt->triangle_count = 0;

    VkBuffer aabb_bufs[3] = { aabb_buf, VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t counts[3] = { static_cast<uint32_t>(sphere_count), 0, 0 };
    bool ok = build_primitive_as(ctx, aabb_bufs, counts, 1, rt);

    vmaDestroyBuffer(ctx->allocator, aabb_buf, aabb_alloc);

    if (!ok) {
        vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
        delete rt;
        PyErr_SetString(PyExc_RuntimeError, "Acceleration structure build failed");
        return NULL;
    }
    VK_TRACE("RT build_sphere_accel done rt=%p tlas=%p", (void*)rt, (void*)rt->tlas);
    ctx->live_accel_structs.insert(rt);

    // Return (rt_handle, sphere_buffer_handle) to mirror the Metal API. The
    // sphere-buffer pointer is informational for the Python side; the C++
    // dispatch uses rt->sphere_buffer directly.
    return Py_BuildValue("nn", reinterpret_cast<Py_ssize_t>(rt),
                                reinterpret_cast<Py_ssize_t>(rt->sphere_buffer));
}

// ---------------------------------------------------------------------------
//  build_mixed_accel — sphere geometry (id 0) + cylinder geometry (id 1).
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_build_mixed_accel(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    Py_buffer sphere_buf, cyl_buf;
    int sphere_count, cyl_count;
    static const char* kwlist[] = {"context", "sphere_data", "sphere_count",
                                   "cylinder_data", "cylinder_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ny*iy*i"), (char**)kwlist,
            &ctx_ptr, &sphere_buf, &sphere_count, &cyl_buf, &cyl_count))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("RT build_mixed_accel sph=%d cyl=%d", sphere_count, cyl_count);
    if (!ctx->raytracing_supported) {
        PyBuffer_Release(&sphere_buf);
        PyBuffer_Release(&cyl_buf);
        PyErr_SetString(PyExc_RuntimeError, "Vulkan device does not support ray tracing");
        return NULL;
    }

    VkBuffer sphere_data_buf = VK_NULL_HANDLE;
    VmaAllocation sphere_data_alloc = VK_NULL_HANDLE;
    if (!create_host_buffer(ctx, sphere_buf.buf,
            sphere_count * sizeof(float) * 4,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            &sphere_data_buf, &sphere_data_alloc)) {
        PyBuffer_Release(&sphere_buf);
        PyBuffer_Release(&cyl_buf);
        PyErr_SetString(PyExc_RuntimeError, "Sphere data buffer allocation failed");
        return NULL;
    }

    VkBuffer cyl_data_buf = VK_NULL_HANDLE;
    VmaAllocation cyl_data_alloc = VK_NULL_HANDLE;
    if (cyl_count > 0) {
        if (!create_host_buffer(ctx, cyl_buf.buf,
                cyl_count * sizeof(float) * 8,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &cyl_data_buf, &cyl_data_alloc)) {
            vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
            PyBuffer_Release(&sphere_buf);
            PyBuffer_Release(&cyl_buf);
            PyErr_SetString(PyExc_RuntimeError, "Cylinder data buffer allocation failed");
            return NULL;
        }
    }

    std::vector<VkAabbPositionsKHR> sph_aabbs(sphere_count);
    fill_sphere_aabbs(reinterpret_cast<const float*>(sphere_buf.buf),
                      sphere_count, sph_aabbs.data());
    PyBuffer_Release(&sphere_buf);

    std::vector<VkAabbPositionsKHR> cyl_aabbs(cyl_count);
    if (cyl_count > 0)
        fill_cylinder_aabbs(reinterpret_cast<const float*>(cyl_buf.buf),
                            cyl_count, cyl_aabbs.data());
    PyBuffer_Release(&cyl_buf);

    VkBuffer sph_aabb_buf = VK_NULL_HANDLE;
    VmaAllocation sph_aabb_alloc = VK_NULL_HANDLE;
    if (!create_host_buffer(ctx, sph_aabbs.data(),
            sph_aabbs.size() * sizeof(VkAabbPositionsKHR),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            &sph_aabb_buf, &sph_aabb_alloc)) {
        vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
        if (cyl_data_buf)
            vmaDestroyBuffer(ctx->allocator, cyl_data_buf, cyl_data_alloc);
        PyErr_SetString(PyExc_RuntimeError, "Sphere AABB buffer allocation failed");
        return NULL;
    }

    VkBuffer cyl_aabb_buf = VK_NULL_HANDLE;
    VmaAllocation cyl_aabb_alloc = VK_NULL_HANDLE;
    if (cyl_count > 0) {
        if (!create_host_buffer(ctx, cyl_aabbs.data(),
                cyl_aabbs.size() * sizeof(VkAabbPositionsKHR),
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                &cyl_aabb_buf, &cyl_aabb_alloc)) {
            vmaDestroyBuffer(ctx->allocator, sph_aabb_buf, sph_aabb_alloc);
            vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
            vmaDestroyBuffer(ctx->allocator, cyl_data_buf, cyl_data_alloc);
            PyErr_SetString(PyExc_RuntimeError, "Cylinder AABB buffer allocation failed");
            return NULL;
        }
    }

    VkRtAccelStructure* rt = new VkRtAccelStructure();
    rt->ctx = ctx;
    rt->sphere_buffer = sphere_data_buf;
    rt->sphere_buffer_alloc = sphere_data_alloc;
    rt->sphere_count = static_cast<uint32_t>(sphere_count);
    rt->cylinder_buffer = cyl_data_buf;
    rt->cylinder_buffer_alloc = cyl_data_alloc;
    rt->cylinder_count = static_cast<uint32_t>(cyl_count);
    rt->triangle_buffer = VK_NULL_HANDLE;
    rt->triangle_buffer_alloc = VK_NULL_HANDLE;
    rt->triangle_count = 0;

    VkBuffer aabb_bufs[3] = { sph_aabb_buf, cyl_aabb_buf, VK_NULL_HANDLE };
    uint32_t counts[3] = { static_cast<uint32_t>(sphere_count),
                           static_cast<uint32_t>(cyl_count), 0 };
    bool ok = build_primitive_as(ctx, aabb_bufs, counts,
                                 cyl_count > 0 ? 2 : 1, rt);

    vmaDestroyBuffer(ctx->allocator, sph_aabb_buf, sph_aabb_alloc);
    if (cyl_aabb_buf)
        vmaDestroyBuffer(ctx->allocator, cyl_aabb_buf, cyl_aabb_alloc);

    if (!ok) {
        vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
        if (cyl_data_buf)
            vmaDestroyBuffer(ctx->allocator, cyl_data_buf, cyl_data_alloc);
        delete rt;
        PyErr_SetString(PyExc_RuntimeError, "Acceleration structure build failed");
        return NULL;
    }

    ctx->live_accel_structs.insert(rt);
    // Mirror the Metal API: (rt_handle, sphere_buffer, cylinder_buffer).
    return Py_BuildValue("nnn",
        reinterpret_cast<Py_ssize_t>(rt),
        reinterpret_cast<Py_ssize_t>(rt->sphere_buffer),
        reinterpret_cast<Py_ssize_t>(rt->cylinder_buffer));
}

// ---------------------------------------------------------------------------
//  build_scene_accel — sphere (id 0) + cylinder (id 1) + triangle (id 2).
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_build_scene_accel(PyObject* self, PyObject* args, PyObject* kwds)
{
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

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("RT build_scene_accel sph=%d cyl=%d tri=%d",
             sphere_count, cyl_count, tri_count);
    if (!ctx->raytracing_supported) {
        PyBuffer_Release(&sphere_buf);
        PyBuffer_Release(&cyl_buf);
        PyBuffer_Release(&tri_buf);
        PyErr_SetString(PyExc_RuntimeError,
                        "Vulkan device does not support ray tracing");
        return NULL;
    }

    // -- Data buffers (kept alive for the shader to read) --
    VkBuffer sphere_data_buf = VK_NULL_HANDLE;
    VmaAllocation sphere_data_alloc = VK_NULL_HANDLE;
    if (!create_host_buffer(ctx, sphere_buf.buf,
            sphere_count * sizeof(float) * 4,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            &sphere_data_buf, &sphere_data_alloc)) {
        PyBuffer_Release(&sphere_buf);
        PyBuffer_Release(&cyl_buf);
        PyBuffer_Release(&tri_buf);
        PyErr_SetString(PyExc_RuntimeError, "Sphere data buffer alloc failed");
        return NULL;
    }

    VkBuffer cyl_data_buf = VK_NULL_HANDLE;
    VmaAllocation cyl_data_alloc = VK_NULL_HANDLE;
    if (cyl_count > 0) {
        if (!create_host_buffer(ctx, cyl_buf.buf,
                cyl_count * sizeof(float) * 8,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &cyl_data_buf, &cyl_data_alloc)) {
            vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
            PyBuffer_Release(&sphere_buf);
            PyBuffer_Release(&cyl_buf);
            PyBuffer_Release(&tri_buf);
            PyErr_SetString(PyExc_RuntimeError, "Cylinder data buffer alloc failed");
            return NULL;
        }
    }

    VkBuffer tri_data_buf = VK_NULL_HANDLE;
    VmaAllocation tri_data_alloc = VK_NULL_HANDLE;
    if (tri_count > 0) {
        if (!create_host_buffer(ctx, tri_buf.buf,
                tri_count * sizeof(float) * 12,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &tri_data_buf, &tri_data_alloc)) {
            vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
            if (cyl_data_buf)
                vmaDestroyBuffer(ctx->allocator, cyl_data_buf, cyl_data_alloc);
            PyBuffer_Release(&sphere_buf);
            PyBuffer_Release(&cyl_buf);
            PyBuffer_Release(&tri_buf);
            PyErr_SetString(PyExc_RuntimeError,
                            "Triangle data buffer alloc failed");
            return NULL;
        }
    }

    // -- AABBs for each geometry --
    std::vector<VkAabbPositionsKHR> sph_aabbs(sphere_count);
    fill_sphere_aabbs(reinterpret_cast<const float*>(sphere_buf.buf),
                      sphere_count, sph_aabbs.data());
    PyBuffer_Release(&sphere_buf);

    std::vector<VkAabbPositionsKHR> cyl_aabbs(cyl_count);
    if (cyl_count > 0)
        fill_cylinder_aabbs(reinterpret_cast<const float*>(cyl_buf.buf),
                            cyl_count, cyl_aabbs.data());
    PyBuffer_Release(&cyl_buf);

    std::vector<VkAabbPositionsKHR> tri_aabbs(tri_count);
    if (tri_count > 0)
        fill_triangle_aabbs(reinterpret_cast<const float*>(tri_buf.buf),
                            tri_count, tri_aabbs.data());
    PyBuffer_Release(&tri_buf);

    // Create AABB GPU buffers.
    auto make_aabb_buf = [&](auto& vec, VkBuffer* buf, VmaAllocation* alloc) -> bool {
        if (vec.empty()) return true;
        return create_host_buffer(ctx, vec.data(),
            vec.size() * sizeof(VkAabbPositionsKHR),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            buf, alloc);
    };

    VkBuffer sph_aabb_buf = VK_NULL_HANDLE;
    VmaAllocation sph_aabb_alloc = VK_NULL_HANDLE;
    VkBuffer cyl_aabb_buf = VK_NULL_HANDLE;
    VmaAllocation cyl_aabb_alloc = VK_NULL_HANDLE;
    VkBuffer tri_aabb_buf = VK_NULL_HANDLE;
    VmaAllocation tri_aabb_alloc = VK_NULL_HANDLE;

    if (!make_aabb_buf(sph_aabbs, &sph_aabb_buf, &sph_aabb_alloc) ||
        !make_aabb_buf(cyl_aabbs, &cyl_aabb_buf, &cyl_aabb_alloc) ||
        !make_aabb_buf(tri_aabbs, &tri_aabb_buf, &tri_aabb_alloc)) {
        if (sph_aabb_buf) vmaDestroyBuffer(ctx->allocator, sph_aabb_buf, sph_aabb_alloc);
        if (cyl_aabb_buf) vmaDestroyBuffer(ctx->allocator, cyl_aabb_buf, cyl_aabb_alloc);
        if (tri_aabb_buf) vmaDestroyBuffer(ctx->allocator, tri_aabb_buf, tri_aabb_alloc);
        vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
        if (cyl_data_buf) vmaDestroyBuffer(ctx->allocator, cyl_data_buf, cyl_data_alloc);
        if (tri_data_buf) vmaDestroyBuffer(ctx->allocator, tri_data_buf, tri_data_alloc);
        PyErr_SetString(PyExc_RuntimeError, "AABB buffer allocation failed");
        return NULL;
    }

    VkRtAccelStructure* rt = new VkRtAccelStructure();
    rt->ctx = ctx;
    rt->sphere_buffer = sphere_data_buf;
    rt->sphere_buffer_alloc = sphere_data_alloc;
    rt->sphere_count = static_cast<uint32_t>(sphere_count);
    rt->cylinder_buffer = cyl_data_buf;
    rt->cylinder_buffer_alloc = cyl_data_alloc;
    rt->cylinder_count = static_cast<uint32_t>(cyl_count);
    rt->triangle_buffer = tri_data_buf;
    rt->triangle_buffer_alloc = tri_data_alloc;
    rt->triangle_count = static_cast<uint32_t>(tri_count);

    uint32_t geom_count = 1;
    if (cyl_count > 0) geom_count = 2;
    if (tri_count > 0) geom_count = 3;

    // If there are triangles but no cylinders, geometry slot 1 is still
    // declared (with 0 primitives) so geometry IDs stay stable. Its AABB
    // data address must nonetheless be valid — vkGetBufferDeviceAddress on
    // VK_NULL_HANDLE is a valid-usage violation — so back the empty slot
    // with a single degenerate AABB.
    if (tri_count > 0 && cyl_count == 0) {
        VkAabbPositionsKHR empty{};
        if (!create_host_buffer(ctx, &empty, sizeof(empty),
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                &cyl_aabb_buf, &cyl_aabb_alloc)) {
            if (sph_aabb_buf) vmaDestroyBuffer(ctx->allocator, sph_aabb_buf, sph_aabb_alloc);
            if (tri_aabb_buf) vmaDestroyBuffer(ctx->allocator, tri_aabb_buf, tri_aabb_alloc);
            vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
            if (tri_data_buf) vmaDestroyBuffer(ctx->allocator, tri_data_buf, tri_data_alloc);
            delete rt;
            PyErr_SetString(PyExc_RuntimeError, "AABB buffer allocation failed");
            return NULL;
        }
    }

    VkBuffer aabb_bufs[3] = { sph_aabb_buf, cyl_aabb_buf, tri_aabb_buf };
    uint32_t counts[3] = { static_cast<uint32_t>(sphere_count),
                           static_cast<uint32_t>(cyl_count),
                           static_cast<uint32_t>(tri_count) };
    bool ok = build_primitive_as(ctx, aabb_bufs, counts, geom_count, rt);

    if (sph_aabb_buf) vmaDestroyBuffer(ctx->allocator, sph_aabb_buf, sph_aabb_alloc);
    if (cyl_aabb_buf) vmaDestroyBuffer(ctx->allocator, cyl_aabb_buf, cyl_aabb_alloc);
    if (tri_aabb_buf) vmaDestroyBuffer(ctx->allocator, tri_aabb_buf, tri_aabb_alloc);

    if (!ok) {
        vmaDestroyBuffer(ctx->allocator, sphere_data_buf, sphere_data_alloc);
        if (cyl_data_buf) vmaDestroyBuffer(ctx->allocator, cyl_data_buf, cyl_data_alloc);
        if (tri_data_buf) vmaDestroyBuffer(ctx->allocator, tri_data_buf, tri_data_alloc);
        delete rt;
        PyErr_SetString(PyExc_RuntimeError, "Acceleration structure build failed");
        return NULL;
    }

    ctx->live_accel_structs.insert(rt);
    return Py_BuildValue("nnnn",
        reinterpret_cast<Py_ssize_t>(rt),
        reinterpret_cast<Py_ssize_t>(rt->sphere_buffer),
        reinterpret_cast<Py_ssize_t>(rt->cylinder_buffer),
        reinterpret_cast<Py_ssize_t>(rt->triangle_buffer));
}

// ---------------------------------------------------------------------------
//  destroy_accel_struct — frees the AS, its backing buffer, and per-primitive
//  buffers. The sphere_buffer arg is ignored (kept for API parity with Metal,
//  which returns it as a separate handle); the buffer is freed via the rt
//  struct it belongs to.
// ---------------------------------------------------------------------------
static void destroy_accel_struct_internal(VulkanContext* ctx,
                                          VkRtAccelStructure* rt)
{
    if (rt->tlas != VK_NULL_HANDLE)
        ctx->pfn_destroy_as(ctx->device, rt->tlas, nullptr);
    if (rt->tlas_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx->allocator, rt->tlas_buffer, rt->tlas_buffer_alloc);
    if (rt->instance_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx->allocator, rt->instance_buffer, rt->instance_buffer_alloc);
    if (rt->blas != VK_NULL_HANDLE)
        ctx->pfn_destroy_as(ctx->device, rt->blas, nullptr);
    if (rt->blas_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx->allocator, rt->blas_buffer, rt->blas_buffer_alloc);
    if (rt->sphere_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx->allocator, rt->sphere_buffer, rt->sphere_buffer_alloc);
    if (rt->cylinder_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx->allocator, rt->cylinder_buffer, rt->cylinder_buffer_alloc);
    if (rt->triangle_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx->allocator, rt->triangle_buffer, rt->triangle_buffer_alloc);
    delete rt;
}

extern "C"
PyObject* vulkan_destroy_accel_struct(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t as_ptr, sb_ptr;
    static const char* kwlist[] = {"accel_struct", "sphere_buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist,
            &as_ptr, &sb_ptr))
        return NULL;
    (void)sb_ptr;

    if (!as_ptr) Py_RETURN_NONE;
    VkRtAccelStructure* rt = reinterpret_cast<VkRtAccelStructure*>(as_ptr);
    VulkanContext* ctx = rt->ctx;

    // Wait for any in-flight frames to finish before tearing down the AS,
    // since RTAO dispatches in the previous frame may still reference it.
    vkDeviceWaitIdle(ctx->device);

    if (ctx->live_accel_structs.erase(rt) == 0)
        Py_RETURN_NONE;
    destroy_accel_struct_internal(ctx, rt);

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  RTAO compute pipeline — descriptor layout + pipeline creation.
//
//  Bindings (set 0):
//    0  uniform buffer  RTAOUniforms
//    1  acceleration structure
//    2  storage buffer  spheres (vec4)
//    3  combined image  depth texture (sampled, R32 sfloat)
//    4  storage image   ao output       (RGBA8 unorm)
//    5  combined image  prev accumulation (R32 sfloat, sampled)
//    6  storage image   accum output    (R32 sfloat)
// ---------------------------------------------------------------------------
static bool create_rtao_descriptor_layout(VulkanContext* ctx)
{
    VkDescriptorSetLayoutBinding b[7] = {};
    for (int i = 0; i < 7; i++) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 7;
    ci.pBindings = b;
    return vkCreateDescriptorSetLayout(ctx->device, &ci, nullptr,
                                       &s_rtao.set_layout) == VK_SUCCESS;
}

static bool create_rtao_descriptor_pool(VulkanContext* ctx)
{
    // Just enough for one set per frame-in-flight slot. Sets are
    // re-written each dispatch (after the fence wait), never freed.
    constexpr uint32_t SETS = MAX_FRAMES_IN_FLIGHT;
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,           SETS },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, SETS },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,           SETS },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2*SETS },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          2*SETS },
    };
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = SETS;
    ci.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
    ci.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx->device, &ci, nullptr,
                               &s_rtao.descriptor_pool) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayout layouts[MAX_FRAMES_IN_FLIGHT];
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        layouts[i] = s_rtao.set_layout;

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = s_rtao.descriptor_pool;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts = layouts;
    return vkAllocateDescriptorSets(ctx->device, &ai, s_rtao.sets) == VK_SUCCESS;
}

extern "C"
PyObject* vulkan_compile_rtao(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    const char* shader_source;
    static const char* kwlist[] = {"context", "shader_source", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ns"), (char**)kwlist,
            &ctx_ptr, &shader_source))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("RT compile_rtao");
    if (!ctx->raytracing_supported) {
        PyErr_SetString(PyExc_RuntimeError, "Vulkan device does not support ray tracing");
        return NULL;
    }

    if (s_rtao_compiled) { VK_TRACE("  already compiled"); Py_RETURN_NONE; }

    // ---- Descriptor + pipeline layout ----
    if (!create_rtao_descriptor_layout(ctx)) {
        PyErr_SetString(PyExc_RuntimeError, "RTAO descriptor layout failed");
        return NULL;
    }
    if (!create_rtao_descriptor_pool(ctx)) {
        destroy_rtao_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "RTAO descriptor pool failed");
        return NULL;
    }

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &s_rtao.set_layout;
    if (vkCreatePipelineLayout(ctx->device, &pli, nullptr,
                               &s_rtao.pipeline_layout) != VK_SUCCESS) {
        destroy_rtao_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "RTAO pipeline layout failed");
        return NULL;
    }

    // ---- Shader compilation (GLSL -> SPIR-V via shaderc) ----
    // Target Vulkan 1.2 so VK_KHR_ray_query / SPV_KHR_ray_query are in scope.
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    shaderc_compile_options_t opts = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan,
                                           shaderc_env_version_vulkan_1_2);
    shaderc_compile_options_set_target_spirv(opts, shaderc_spirv_version_1_4);
    shaderc_compile_options_set_optimization_level(opts,
        shaderc_optimization_level_performance);

    shaderc_compilation_result_t res = shaderc_compile_into_spv(
        compiler, shader_source, strlen(shader_source),
        shaderc_glsl_compute_shader, "rtao.comp", "main", opts);

    if (shaderc_result_get_compilation_status(res) !=
        shaderc_compilation_status_success) {
        PyErr_Format(PyExc_RuntimeError, "RTAO shader compile failed: %s",
                     shaderc_result_get_error_message(res));
        shaderc_result_release(res);
        shaderc_compile_options_release(opts);
        shaderc_compiler_release(compiler);
        destroy_rtao_pipeline_state(ctx);
        return NULL;
    }

    VkShaderModule mod = create_shader_module(ctx,
        reinterpret_cast<const uint32_t*>(shaderc_result_get_bytes(res)),
        shaderc_result_get_length(res));
    shaderc_result_release(res);
    shaderc_compile_options_release(opts);
    shaderc_compiler_release(compiler);
    if (mod == VK_NULL_HANDLE) {
        destroy_rtao_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "RTAO shader module creation failed");
        return NULL;
    }

    // ---- Compute pipeline ----
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = s_rtao.pipeline_layout;

    VkResult vr = vkCreateComputePipelines(ctx->device, ctx->pipeline_cache,
                                           1, &cpi, nullptr, &s_rtao.pipeline);
    vkDestroyShaderModule(ctx->device, mod, nullptr);
    if (vr != VK_SUCCESS) {
        destroy_rtao_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "RTAO compute pipeline creation failed");
        return NULL;
    }

    // Sampler used for the depth + prev-accumulation reads. Linear is fine
    // since we only use them for spatial reconstruction / temporal blend.
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    if (vkCreateSampler(ctx->device, &si, nullptr, &s_rtao.sampler) != VK_SUCCESS) {
        destroy_rtao_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "RTAO sampler creation failed");
        return NULL;
    }

    s_rtao_compiled = true;
    VK_TRACE("  compile_rtao OK pipeline=%p", (void*)s_rtao.pipeline);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Helper: layout transition for a compute storage image. Takes the current
//  layout from the texture wrapper and updates it after the transition.
// ---------------------------------------------------------------------------
static void transition_for_storage(VkCommandBuffer cmd, VulkanTexture* tex)
{
    if (tex->current_layout == VK_IMAGE_LAYOUT_GENERAL) return;
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = tex->current_layout;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = tex->image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b);
    tex->current_layout = VK_IMAGE_LAYOUT_GENERAL;
}

static void transition_for_sampled(VkCommandBuffer cmd, VulkanTexture* tex,
                                   VkImageAspectFlags aspect)
{
    if (tex->current_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return;
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = tex->current_layout;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = tex->image;
    b.subresourceRange.aspectMask = aspect;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT
                    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    // Consumers are both compute (prev-accumulation reads) and the raster
    // composite/denoise passes' fragment shaders; the destination scope
    // must include FRAGMENT_SHADER or the raster pass can sample the image
    // while the compute kernel is still writing it.
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
        | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b);
    tex->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// ---------------------------------------------------------------------------
//  dispatch_rtao — one frame's compute kernel. Allocates a fresh descriptor
//  set, uploads the per-frame uniforms into a transient buffer, transitions
//  textures, dispatches, and transitions back to SHADER_READ for the
//  subsequent denoise + composite passes.
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_dispatch_rtao(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr, as_ptr, sb_ptr;
    Py_ssize_t ao_tex_ptr, prev_tex_ptr, accum_tex_ptr;
    Py_buffer uniforms_buf;
    static const char* kwlist[] = {"context", "accel_struct", "sphere_buffer",
                                   "ao_texture", "prev_texture", "accum_texture",
                                   "uniforms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nnnnnny*"), (char**)kwlist,
            &ctx_ptr, &as_ptr, &sb_ptr, &ao_tex_ptr, &prev_tex_ptr,
            &accum_tex_ptr, &uniforms_buf))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    if (!s_rtao_compiled) {
        PyBuffer_Release(&uniforms_buf);
        PyErr_SetString(PyExc_RuntimeError, "RTAO pipeline not compiled");
        return NULL;
    }
    (void)sb_ptr;  // sphere_buffer is reached via the AS struct.

    VkRtAccelStructure* rt = reinterpret_cast<VkRtAccelStructure*>(as_ptr);
    VulkanTexture* ao_tex = reinterpret_cast<VulkanTexture*>(ao_tex_ptr);
    VulkanTexture* prev_tex = reinterpret_cast<VulkanTexture*>(prev_tex_ptr);
    VulkanTexture* accum_tex = reinterpret_cast<VulkanTexture*>(accum_tex_ptr);
    VK_TRACE("RT dispatch_rtao rt=%p ao=%p (%ux%u, layout=%d) prev=%p accum=%p in_pass=%d",
             (void*)rt, (void*)ao_tex, ao_tex->width, ao_tex->height,
             (int)ao_tex->current_layout, (void*)prev_tex, (void*)accum_tex,
             ctx->in_render_pass ? 1 : 0);

    // ---- Per-frame uniform buffer ----
    // Created host-visible and freed at end of frame via the deferred
    // delete queue, since the GPU may still read it.
    BufferWithAllocator* ubo = new BufferWithAllocator();
    ubo->ctx = ctx;
    ubo->allocator = ctx->allocator;
    if (!create_host_buffer(ctx, uniforms_buf.buf, uniforms_buf.len,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            &ubo->base.buffer, &ubo->base.allocation)) {
        PyBuffer_Release(&uniforms_buf);
        delete ubo;
        PyErr_SetString(PyExc_RuntimeError, "RTAO uniform buffer alloc failed");
        return NULL;
    }
    ubo->base.size = uniforms_buf.len;
    PyBuffer_Release(&uniforms_buf);

    // Use this frame slot's pre-allocated descriptor set. begin_frame
    // already waited on the fence for this slot, so it's safe to rewrite.
    VkDescriptorSet set = s_rtao.sets[ctx->current_frame];

    // ---- Wire up the descriptor set ----
    VkDescriptorBufferInfo ubo_info{};
    ubo_info.buffer = ubo->base.buffer;
    ubo_info.offset = 0;
    ubo_info.range = ubo->base.size;

    VkWriteDescriptorSetAccelerationStructureKHR as_write{};
    as_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_write.accelerationStructureCount = 1;
    as_write.pAccelerationStructures = &rt->tlas;

    VkDescriptorBufferInfo sphere_info{};
    sphere_info.buffer = rt->sphere_buffer;
    sphere_info.offset = 0;
    sphere_info.range = VK_WHOLE_SIZE;

    // Use the depth attachment of the active target. High-precision image
    // rendering and software stereo render into an offscreen target while
    // the same RTAO pass is active.
    OffscreenFramebuffer* off = ctx->active_offscreen;
    VulkanTexture depth_tex_view = {};  // shim wrapper for current_layout
    depth_tex_view.image = off ? off->depth_image : ctx->depth_image.image;
    depth_tex_view.image_view = off ? off->depth_view : ctx->depth_image.image_view;
    depth_tex_view.format = VK_FORMAT_D32_SFLOAT;
    depth_tex_view.current_layout = off
        ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        : ctx->depth_image.current_layout;

    VkDescriptorImageInfo depth_info{};
    depth_info.sampler = s_rtao.sampler;
    depth_info.imageView = depth_tex_view.image_view;
    depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ao_info{};
    ao_info.sampler = VK_NULL_HANDLE;
    ao_info.imageView = ao_tex->image_view;
    ao_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo prev_info{};
    prev_info.sampler = s_rtao.sampler;
    prev_info.imageView = prev_tex->image_view;
    prev_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo accum_info{};
    accum_info.sampler = VK_NULL_HANDLE;
    accum_info.imageView = accum_tex->image_view;
    accum_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w[7] = {};
    for (int i = 0; i < 7; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &ubo_info;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w[1].pNext = &as_write;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo = &sphere_info;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[3].pImageInfo = &depth_info;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[4].pImageInfo = &ao_info;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[5].pImageInfo = &prev_info;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[6].pImageInfo = &accum_info;
    vkUpdateDescriptorSets(ctx->device, 7, w, 0, nullptr);

    // ---- Layout transitions and dispatch ----
    VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;

    // Depth: from DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> DEPTH_STENCIL_READ_ONLY.
    {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // UNDEFINED means a fresh depth image (e.g. right after a resize)
        // that no pass has touched; claiming any other oldLayout would
        // mismatch the image's actual layout. UNDEFINED-as-old is always
        // valid (contents discarded), and the depth is garbage either way.
        b.oldLayout = depth_tex_view.current_layout;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = depth_tex_view.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
        depth_tex_view.current_layout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        if (!off)
            ctx->depth_image.current_layout = depth_tex_view.current_layout;
    }

    transition_for_storage(cmd, ao_tex);
    transition_for_storage(cmd, accum_tex);
    transition_for_sampled(cmd, prev_tex, VK_IMAGE_ASPECT_COLOR_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_rtao.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            s_rtao.pipeline_layout, 0, 1, &set, 0, nullptr);

    uint32_t gx = (ao_tex->width  + 7) / 8;
    uint32_t gy = (ao_tex->height + 7) / 8;
    VK_TRACE("  dispatch %ux%u groups", gx, gy);
    vkCmdDispatch(cmd, gx, gy, 1);

    // Make AO/accum readable by the subsequent fragment passes.
    transition_for_sampled(cmd, ao_tex, VK_IMAGE_ASPECT_COLOR_BIT);
    transition_for_sampled(cmd, accum_tex, VK_IMAGE_ASPECT_COLOR_BIT);

    // Restore depth layout for the upcoming color pass.
    {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = depth_tex_view.current_layout;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = depth_tex_view.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                        | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
        depth_tex_view.current_layout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        if (!off)
            ctx->depth_image.current_layout = depth_tex_view.current_layout;
    }

    // Defer the uniform buffer until the in-flight frame that uses it
    // retires (deferred_delete_slot keeps inside/outside-frame distinction).
    ctx->pending_deletes[deferred_delete_slot(ctx)].buffers.push_back(ubo);

    Py_RETURN_NONE;
}

// ===========================================================================
//  Path tracer
// ===========================================================================
//
//  Bindings (set 0):
//    0  uniform PTUniforms
//    1  acceleration structure (TLAS)
//    2  storage  spheres (vec4)
//    3  storage  cylinders (2x vec4 each)
//    4  storage  sphere_colors  (uchar4 packed as uint)
//    5  storage  cylinder_colors (uchar4 packed as uint)
//    6  storage image color_tex  (RGBA8 unorm, write)
//    7  combined image  prev_tex (RGBA8 unorm, sampled)
//    8  storage image accum_tex  (RGBA8 unorm, write)
//    9  storage  triangles (3x vec4 each)
//   10  storage  triangle_colors (uchar4 packed as uint)
//   11  combined image  vol_density  (sampler3D; 1x1x1 dummy when no volume)
//   12  combined image  vol_colormap (sampler1D; 1-texel dummy when no volume)
// ---------------------------------------------------------------------------

static bool create_pt_descriptor_layout(VulkanContext* ctx)
{
    VkDescriptorSetLayoutBinding b[13] = {};
    for (int i = 0; i < 13; i++) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 13;
    ci.pBindings = b;
    return vkCreateDescriptorSetLayout(ctx->device, &ci, nullptr,
                                       &s_pt.set_layout) == VK_SUCCESS;
}

static bool create_pt_descriptor_pool(VulkanContext* ctx)
{
    constexpr uint32_t SETS = MAX_FRAMES_IN_FLIGHT;
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             SETS },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, SETS },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,           6*SETS },  // sph+cyl+sphcol+cylcol+tri+tricol
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,   3*SETS },  // prev_tex + vol_density + vol_colormap
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            2*SETS },
    };
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = SETS;
    ci.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
    ci.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx->device, &ci, nullptr,
                               &s_pt.descriptor_pool) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayout layouts[MAX_FRAMES_IN_FLIGHT];
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        layouts[i] = s_pt.set_layout;
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = s_pt.descriptor_pool;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts = layouts;
    return vkAllocateDescriptorSets(ctx->device, &ai, s_pt.sets) == VK_SUCCESS;
}

extern "C"
PyObject* vulkan_compile_pathtracer(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    const char* shader_source;
    static const char* kwlist[] = {"context", "shader_source", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ns"), (char**)kwlist,
            &ctx_ptr, &shader_source))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("RT compile_pathtracer");
    if (!ctx->raytracing_supported) {
        PyErr_SetString(PyExc_RuntimeError, "Vulkan device does not support ray tracing");
        return NULL;
    }
    if (s_pt_compiled) { VK_TRACE("  already compiled"); Py_RETURN_NONE; }

    if (!create_pt_descriptor_layout(ctx)) {
        PyErr_SetString(PyExc_RuntimeError, "PT descriptor layout failed");
        return NULL;
    }
    if (!create_pt_descriptor_pool(ctx)) {
        destroy_pt_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "PT descriptor pool failed");
        return NULL;
    }

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &s_pt.set_layout;
    if (vkCreatePipelineLayout(ctx->device, &pli, nullptr,
                               &s_pt.pipeline_layout) != VK_SUCCESS) {
        destroy_pt_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "PT pipeline layout failed");
        return NULL;
    }

    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    shaderc_compile_options_t opts = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan,
                                           shaderc_env_version_vulkan_1_2);
    shaderc_compile_options_set_target_spirv(opts, shaderc_spirv_version_1_4);
    shaderc_compile_options_set_optimization_level(opts,
        shaderc_optimization_level_performance);

    shaderc_compilation_result_t res = shaderc_compile_into_spv(
        compiler, shader_source, strlen(shader_source),
        shaderc_glsl_compute_shader, "pathtracer.comp", "main", opts);

    if (shaderc_result_get_compilation_status(res) !=
        shaderc_compilation_status_success) {
        PyErr_Format(PyExc_RuntimeError, "Path-tracer shader compile failed: %s",
                     shaderc_result_get_error_message(res));
        shaderc_result_release(res);
        shaderc_compile_options_release(opts);
        shaderc_compiler_release(compiler);
        destroy_pt_pipeline_state(ctx);
        return NULL;
    }

    VkShaderModule mod = create_shader_module(ctx,
        reinterpret_cast<const uint32_t*>(shaderc_result_get_bytes(res)),
        shaderc_result_get_length(res));
    shaderc_result_release(res);
    shaderc_compile_options_release(opts);
    shaderc_compiler_release(compiler);
    if (mod == VK_NULL_HANDLE) {
        destroy_pt_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "PT shader module creation failed");
        return NULL;
    }

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = s_pt.pipeline_layout;
    VkResult vr = vkCreateComputePipelines(ctx->device, ctx->pipeline_cache,
                                           1, &cpi, nullptr, &s_pt.pipeline);
    vkDestroyShaderModule(ctx->device, mod, nullptr);
    if (vr != VK_SUCCESS) {
        destroy_pt_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "PT compute pipeline creation failed");
        return NULL;
    }

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    if (vkCreateSampler(ctx->device, &si, nullptr, &s_pt.sampler) != VK_SUCCESS) {
        destroy_pt_pipeline_state(ctx);
        PyErr_SetString(PyExc_RuntimeError, "PT sampler creation failed");
        return NULL;
    }

    // Dummy zeroed storage buffer used as a fallback for cylinder + cyl
    // color bindings when a dispatch passes 0. Big enough for one entry
    // (32 bytes covers either a Cylinder or a uint with padding).
    {
        uint8_t zeros[32] = {};
        if (!create_host_buffer(ctx, zeros, sizeof(zeros),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                &s_pt.dummy_buffer, &s_pt.dummy_buffer_alloc)) {
            destroy_pt_pipeline_state(ctx);
            PyErr_SetString(PyExc_RuntimeError,
                            "PT dummy buffer creation failed");
            return NULL;
        }
    }

    s_pt_compiled = true;
    VK_TRACE("  compile_pathtracer OK pipeline=%p", (void*)s_pt.pipeline);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_dispatch_pathtracer(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr, as_ptr, sb_ptr;
    Py_ssize_t color_tex_ptr, prev_tex_ptr, accum_tex_ptr;
    Py_ssize_t color_buf_ptr, cyl_buf_ptr, cyl_color_buf_ptr;
    Py_ssize_t tri_buf_ptr = 0, tri_color_buf_ptr = 0;
    Py_ssize_t vol_density_tex_ptr = 0, vol_colormap_tex_ptr = 0;
    Py_buffer uniforms_buf;
    static const char* kwlist[] = {"context", "accel_struct", "sphere_buffer",
                                   "color_texture", "prev_texture", "accum_texture",
                                   "color_buffer", "cylinder_buffer",
                                   "cylinder_color_buffer",
                                   "triangle_buffer", "triangle_color_buffer",
                                   "volume_density_texture",
                                   "volume_colormap_texture",
                                   "uniforms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nnnnnnnnnnnnny*"), (char**)kwlist,
            &ctx_ptr, &as_ptr, &sb_ptr, &color_tex_ptr, &prev_tex_ptr,
            &accum_tex_ptr, &color_buf_ptr, &cyl_buf_ptr, &cyl_color_buf_ptr,
            &tri_buf_ptr, &tri_color_buf_ptr,
            &vol_density_tex_ptr, &vol_colormap_tex_ptr, &uniforms_buf))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    if (!s_pt_compiled) {
        PyBuffer_Release(&uniforms_buf);
        PyErr_SetString(PyExc_RuntimeError, "Path tracer pipeline not compiled");
        return NULL;
    }
    (void)sb_ptr;

    VkRtAccelStructure* rt = reinterpret_cast<VkRtAccelStructure*>(as_ptr);
    VulkanTexture* color_tex = reinterpret_cast<VulkanTexture*>(color_tex_ptr);
    VulkanTexture* prev_tex  = reinterpret_cast<VulkanTexture*>(prev_tex_ptr);
    VulkanTexture* accum_tex = reinterpret_cast<VulkanTexture*>(accum_tex_ptr);
    // Volume density (sampler3D) + transfer-function colormap (sampler1D).
    // The host always passes valid handles (1-texel dummies when no volume
    // is shown), so the descriptor set is always complete.
    VulkanTexture* vol_density_tex  = reinterpret_cast<VulkanTexture*>(vol_density_tex_ptr);
    VulkanTexture* vol_colormap_tex = reinterpret_cast<VulkanTexture*>(vol_colormap_tex_ptr);
    BufferWithAllocator* color_buf = resolve_buffer_handle(color_buf_ptr);
    if (!color_buf) {
        PyBuffer_Release(&uniforms_buf);
        return NULL;
    }
    VK_TRACE("RT dispatch_pathtracer rt=%p color=%p (%ux%u, layout=%d) cyl_buf=%lld in_pass=%d",
             (void*)rt, (void*)color_tex, color_tex->width, color_tex->height,
             (int)color_tex->current_layout,
             (long long)cyl_buf_ptr, ctx->in_render_pass ? 1 : 0);

    BufferWithAllocator* cyl_color_buf = cyl_color_buf_ptr
        ? resolve_buffer_handle(cyl_color_buf_ptr) : nullptr;
    if (cyl_color_buf_ptr && !cyl_color_buf) {
        PyBuffer_Release(&uniforms_buf);
        return NULL;
    }
    BufferWithAllocator* tri_color_buf = tri_color_buf_ptr
        ? resolve_buffer_handle(tri_color_buf_ptr) : nullptr;
    if (tri_color_buf_ptr && !tri_color_buf) {
        PyBuffer_Release(&uniforms_buf);
        return NULL;
    }

    // Per-frame uniform buffer (defer-freed at end of frame).
    BufferWithAllocator* ubo = new BufferWithAllocator();
    ubo->ctx = ctx;
    ubo->allocator = ctx->allocator;
    if (!create_host_buffer(ctx, uniforms_buf.buf, uniforms_buf.len,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            &ubo->base.buffer, &ubo->base.allocation)) {
        PyBuffer_Release(&uniforms_buf);
        delete ubo;
        PyErr_SetString(PyExc_RuntimeError, "PT uniform buffer alloc failed");
        return NULL;
    }
    ubo->base.size = uniforms_buf.len;
    PyBuffer_Release(&uniforms_buf);

    VkDescriptorSet set = s_pt.sets[ctx->current_frame];

    // Pick buffer fallbacks. Spheres are always present; cylinders and
    // triangles may be missing.
    VkBuffer cyl_geo_buf = (cyl_buf_ptr != 0)
        ? reinterpret_cast<VkBuffer>(cyl_buf_ptr)
        : s_pt.dummy_buffer;
    VkBuffer cyl_col_buf = cyl_color_buf
        ? cyl_color_buf->base.buffer
        : s_pt.dummy_buffer;
    VkBuffer tri_geo_buf = (tri_buf_ptr != 0)
        ? reinterpret_cast<VkBuffer>(tri_buf_ptr)
        : s_pt.dummy_buffer;
    VkBuffer tri_col_buf = tri_color_buf
        ? tri_color_buf->base.buffer
        : s_pt.dummy_buffer;

    VkDescriptorBufferInfo ubo_info{};
    ubo_info.buffer = ubo->base.buffer;
    ubo_info.range  = ubo->base.size;

    VkWriteDescriptorSetAccelerationStructureKHR as_write{};
    as_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_write.accelerationStructureCount = 1;
    as_write.pAccelerationStructures = &rt->tlas;

    VkDescriptorBufferInfo sphere_info{};
    sphere_info.buffer = rt->sphere_buffer;
    sphere_info.range  = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo cyl_info{};
    cyl_info.buffer = cyl_geo_buf;
    cyl_info.range  = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo sph_col_info{};
    sph_col_info.buffer = color_buf->base.buffer;
    sph_col_info.range  = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo cyl_col_info{};
    cyl_col_info.buffer = cyl_col_buf;
    cyl_col_info.range  = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo tri_info{};
    tri_info.buffer = tri_geo_buf;
    tri_info.range  = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo tri_col_info{};
    tri_col_info.buffer = tri_col_buf;
    tri_col_info.range  = VK_WHOLE_SIZE;

    VkDescriptorImageInfo color_info{};
    color_info.imageView = color_tex->image_view;
    color_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo prev_info{};
    prev_info.sampler = s_pt.sampler;
    prev_info.imageView = prev_tex->image_view;
    prev_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo accum_info{};
    accum_info.imageView = accum_tex->image_view;
    accum_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo vol_density_info{};
    vol_density_info.sampler = s_pt.sampler;
    vol_density_info.imageView = vol_density_tex->image_view;
    vol_density_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo vol_colormap_info{};
    vol_colormap_info.sampler = s_pt.sampler;
    vol_colormap_info.imageView = vol_colormap_tex->image_view;
    vol_colormap_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w[13] = {};
    for (int i = 0; i < 13; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;          w[0].pBufferInfo = &ubo_info;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; w[1].pNext = &as_write;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;          w[2].pBufferInfo = &sphere_info;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;          w[3].pBufferInfo = &cyl_info;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;          w[4].pBufferInfo = &sph_col_info;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;          w[5].pBufferInfo = &cyl_col_info;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;           w[6].pImageInfo  = &color_info;
    w[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;  w[7].pImageInfo  = &prev_info;
    w[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;           w[8].pImageInfo  = &accum_info;
    w[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;          w[9].pBufferInfo = &tri_info;
    w[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[10].pBufferInfo = &tri_col_info;
    w[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[11].pImageInfo  = &vol_density_info;
    w[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[12].pImageInfo  = &vol_colormap_info;
    vkUpdateDescriptorSets(ctx->device, 13, w, 0, nullptr);

    VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;
    transition_for_storage(cmd, color_tex);
    transition_for_storage(cmd, accum_tex);
    transition_for_sampled(cmd, prev_tex, VK_IMAGE_ASPECT_COLOR_BIT);
    transition_for_sampled(cmd, vol_density_tex, VK_IMAGE_ASPECT_COLOR_BIT);
    transition_for_sampled(cmd, vol_colormap_tex, VK_IMAGE_ASPECT_COLOR_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_pt.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            s_pt.pipeline_layout, 0, 1, &set, 0, nullptr);
    uint32_t gx = (color_tex->width  + 7) / 8;
    uint32_t gy = (color_tex->height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Make outputs sampleable again so the blit pass can read color_tex.
    transition_for_sampled(cmd, color_tex, VK_IMAGE_ASPECT_COLOR_BIT);
    transition_for_sampled(cmd, accum_tex, VK_IMAGE_ASPECT_COLOR_BIT);

    // Use deferred_delete_slot (not current_frame directly) so a dispatch
    // recorded outside begin/end_frame still lands on the in-flight slot.
    ctx->pending_deletes[deferred_delete_slot(ctx)].buffers.push_back(ubo);
    Py_RETURN_NONE;
}

// Context teardown entry point.  The caller has already waited for the device
// to become idle, so all RT device children can be released immediately.
void destroy_raytracing_resources(VulkanContext* ctx)
{
    for (void* handle : ctx->live_accel_structs)
        destroy_accel_struct_internal(ctx,
            reinterpret_cast<VkRtAccelStructure*>(handle));
    ctx->live_accel_structs.clear();

    destroy_rtao_pipeline_state(ctx);
    destroy_pt_pipeline_state(ctx);
}

#undef s_rtao
#undef s_rtao_compiled
#undef s_pt
#undef s_pt_compiled
