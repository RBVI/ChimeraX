/*
 * Vulkan buffer management: create, update, destroy GPU buffers.
 * Draw command recording and descriptor set binding.
 */

#include <Python.h>
#include "vulkan_internal.h"

#include <cstring>

// ---------------------------------------------------------------------------
//  Extended buffer struct that carries the VMA allocator handle so that
//  update and destroy can map/free memory without needing the context.
//  The base VulkanBuffer sits at offset 0, so a pointer to this struct
//  can be safely reinterpret_cast to VulkanBuffer*.
// ---------------------------------------------------------------------------
struct BufferWithAllocator {
    VulkanBuffer base;
    VmaAllocator allocator;
};

// ---------------------------------------------------------------------------
//  Buffer create / update / destroy
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_create_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    Py_buffer py_buf;
    static const char* kwlist[] = {"context", "data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ly*"), (char**)kwlist, &ctx_ptr, &py_buf))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    Py_ssize_t data_len = py_buf.len;

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = data_len;
    buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                      | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                      | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBuffer vk_buffer;
    VmaAllocation allocation;
    VkResult result = vmaCreateBuffer(ctx->allocator, &buffer_info, &alloc_info,
                                      &vk_buffer, &allocation, nullptr);
    if (result != VK_SUCCESS) {
        PyBuffer_Release(&py_buf);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Vulkan buffer");
        return NULL;
    }

    // Copy data into the buffer.
    void* mapped = nullptr;
    result = vmaMapMemory(ctx->allocator, allocation, &mapped);
    if (result != VK_SUCCESS) {
        vmaDestroyBuffer(ctx->allocator, vk_buffer, allocation);
        PyBuffer_Release(&py_buf);
        PyErr_SetString(PyExc_RuntimeError, "Failed to map Vulkan buffer memory");
        return NULL;
    }
    memcpy(mapped, py_buf.buf, data_len);
    vmaUnmapMemory(ctx->allocator, allocation);

    PyBuffer_Release(&py_buf);

    BufferWithAllocator* bwa = new BufferWithAllocator();
    bwa->base.buffer = vk_buffer;
    bwa->base.allocation = allocation;
    bwa->base.size = static_cast<VkDeviceSize>(data_len);
    bwa->allocator = ctx->allocator;

    return PyLong_FromLong(reinterpret_cast<long>(bwa));
}

extern "C"
PyObject* vulkan_update_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long buf_ptr;
    Py_buffer py_buf;
    static const char* kwlist[] = {"buffer", "data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ly*"), (char**)kwlist, &buf_ptr, &py_buf))
        return NULL;

    VulkanBuffer* buf = reinterpret_cast<VulkanBuffer*>(buf_ptr);

    if (static_cast<VkDeviceSize>(py_buf.len) > buf->size) {
        PyBuffer_Release(&py_buf);
        PyErr_SetString(PyExc_RuntimeError,
                        "Buffer too small for update; recreate instead");
        return NULL;
    }

    // The allocator handle is stored alongside the VulkanBuffer by
    // vulkan_create_buffer (see BufferWithAllocator).
    BufferWithAllocator* bwa = reinterpret_cast<BufferWithAllocator*>(buf_ptr);

    void* mapped = nullptr;
    VkResult result = vmaMapMemory(bwa->allocator, buf->allocation, &mapped);
    if (result != VK_SUCCESS) {
        PyBuffer_Release(&py_buf);
        PyErr_SetString(PyExc_RuntimeError, "Failed to map Vulkan buffer for update");
        return NULL;
    }
    memcpy(mapped, py_buf.buf, py_buf.len);
    vmaUnmapMemory(bwa->allocator, buf->allocation);

    PyBuffer_Release(&py_buf);
    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_destroy_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long buf_ptr;
    static const char* kwlist[] = {"buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("l"), (char**)kwlist, &buf_ptr))
        return NULL;

    BufferWithAllocator* bwa = reinterpret_cast<BufferWithAllocator*>(buf_ptr);
    VulkanBuffer* buf = &bwa->base;

    vmaDestroyBuffer(bwa->allocator, buf->buffer, buf->allocation);
    delete bwa;

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Draw state binding
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_vertex_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, buf_ptr;
    int binding, offset;
    static const char* kwlist[] = {"context", "buffer", "binding", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llii"), (char**)kwlist,
            &ctx_ptr, &buf_ptr, &binding, &offset))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanBuffer* buf = reinterpret_cast<VulkanBuffer*>(buf_ptr);

    VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;
    VkDeviceSize vk_offset = static_cast<VkDeviceSize>(offset);
    vkCmdBindVertexBuffers(cmd, static_cast<uint32_t>(binding), 1,
                           &buf->buffer, &vk_offset);

    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_set_descriptor_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, buf_ptr;
    int binding, offset;
    static const char* kwlist[] = {"context", "buffer", "binding", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llii"), (char**)kwlist,
            &ctx_ptr, &buf_ptr, &binding, &offset))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanBuffer* buf = reinterpret_cast<VulkanBuffer*>(buf_ptr);

    if (binding < 0 || binding >= static_cast<int>(MAX_UBO_BINDINGS)) {
        PyErr_Format(PyExc_ValueError,
                     "UBO binding %d out of range [0, %u)", binding, MAX_UBO_BINDINGS);
        return NULL;
    }

    ctx->bound_ubo_buffers[binding] = buf;
    ctx->descriptor_dirty = true;

    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_set_fragment_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, tex_ptr;
    int index;
    static const char* kwlist[] = {"context", "texture", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lli"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &index))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanTexture* tex = reinterpret_cast<VulkanTexture*>(tex_ptr);

    if (index < 0 || index >= static_cast<int>(MAX_TEXTURE_BINDINGS)) {
        PyErr_Format(PyExc_ValueError,
                     "Texture index %d out of range [0, %u)", index, MAX_TEXTURE_BINDINGS);
        return NULL;
    }

    ctx->bound_textures[index] = tex;
    ctx->descriptor_dirty = true;

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Descriptor set update (called before draw when dirty)
// ---------------------------------------------------------------------------

static void flush_descriptors(VulkanContext* ctx) {
    if (!ctx->descriptor_dirty)
        return;

    uint32_t frame = ctx->current_frame;
    VkCommandBuffer cmd = ctx->frames[frame].command_buffer;

    // Collect UBO descriptor writes for set 0.
    VkDescriptorBufferInfo buffer_infos[MAX_UBO_BINDINGS] = {};
    VkWriteDescriptorSet writes[MAX_UBO_BINDINGS + MAX_TEXTURE_BINDINGS] = {};
    uint32_t write_count = 0;

    for (uint32_t i = 0; i < MAX_UBO_BINDINGS; i++) {
        if (!ctx->bound_ubo_buffers[i])
            continue;
        VulkanBuffer* buf = ctx->bound_ubo_buffers[i];
        buffer_infos[i].buffer = buf->buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = buf->size;

        VkWriteDescriptorSet& w = writes[write_count++];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ctx->ubo_descriptor_sets[frame];
        w.dstBinding = i;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &buffer_infos[i];
    }

    // Collect texture descriptor writes for set 1.
    VkDescriptorImageInfo image_infos[MAX_TEXTURE_BINDINGS] = {};

    for (uint32_t i = 0; i < MAX_TEXTURE_BINDINGS; i++) {
        if (!ctx->bound_textures[i])
            continue;
        VulkanTexture* tex = ctx->bound_textures[i];
        image_infos[i].sampler = tex->sampler;
        image_infos[i].imageView = tex->image_view;
        image_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet& w = writes[write_count++];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ctx->texture_descriptor_sets[frame];
        w.dstBinding = i;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &image_infos[i];
    }

    if (write_count > 0)
        vkUpdateDescriptorSets(ctx->device, write_count, writes, 0, nullptr);

    // Bind both descriptor sets.
    VkDescriptorSet sets[2] = {
        ctx->ubo_descriptor_sets[frame],
        ctx->texture_descriptor_sets[frame],
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            ctx->shared_pipeline_layout,
                            0, 2, sets, 0, nullptr);

    ctx->descriptor_dirty = false;
}

// ---------------------------------------------------------------------------
//  Draw
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_draw_indexed(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, index_buf_ptr;
    int primitive_type, index_count, index_offset, instance_count;
    static const char* kwlist[] = {"context", "index_buffer", "primitive_type",
                                   "index_count", "index_offset", "instance_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lliiii"), (char**)kwlist,
            &ctx_ptr, &index_buf_ptr, &primitive_type,
            &index_count, &index_offset, &instance_count))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanBuffer* idx_buf = reinterpret_cast<VulkanBuffer*>(index_buf_ptr);

    if (!ctx->in_render_pass)
        Py_RETURN_NONE;

    VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;

    // Flush any pending descriptor set updates.
    flush_descriptors(ctx);

    // Bind the index buffer (always 32-bit unsigned indices).
    VkDeviceSize idx_offset = static_cast<VkDeviceSize>(index_offset) * sizeof(uint32_t);
    vkCmdBindIndexBuffer(cmd, idx_buf->buffer, idx_offset, VK_INDEX_TYPE_UINT32);

    // Issue the draw call.  Primitive topology is baked into the pipeline
    // state, so the primitive_type parameter is not used here.
    (void)primitive_type;
    vkCmdDrawIndexed(cmd,
                     static_cast<uint32_t>(index_count),
                     static_cast<uint32_t>(instance_count),
                     0,   // firstIndex
                     0,   // vertexOffset
                     0);  // firstInstance

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Depth/stencil state
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_depth_stencil_state(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    int depth_write, depth_test, compare_func;
    static const char* kwlist[] = {"context", "depth_write", "depth_test",
                                   "compare_func", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("liii"), (char**)kwlist,
            &ctx_ptr, &depth_write, &depth_test, &compare_func))
        return NULL;

    // In Vulkan 1.0, depth/stencil state is part of the graphics pipeline
    // and cannot be changed dynamically.  We store the desired state so
    // that the pipeline creation or selection logic can use it.
    // When VK_EXT_extended_dynamic_state is available, we can set these
    // dynamically instead.
    (void)ctx_ptr;
    (void)depth_write;
    (void)depth_test;
    (void)compare_func;

    // TODO: Store depth state on context and use it for pipeline selection.
    // For now this is a no-op placeholder.

    Py_RETURN_NONE;
}
