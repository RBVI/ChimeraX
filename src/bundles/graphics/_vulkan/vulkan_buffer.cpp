/*
 * Vulkan buffer management: create, update, destroy GPU buffers.
 * Draw command recording and descriptor set binding.
 */

#include <Python.h>
#include "vulkan_internal.h"

#include <cstring>

// BufferWithAllocator is declared in vulkan_internal.h so the
// deferred-deletion queue can free buffers without a back-reference here.

// ---------------------------------------------------------------------------
//  Buffer create / update / destroy
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_create_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    Py_buffer py_buf;
    static const char* kwlist[] = {"context", "data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ny*"), (char**)kwlist, &ctx_ptr, &py_buf))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    Py_ssize_t data_len = py_buf.len;
    VK_TRACE("create_buffer size=%zd", (Py_ssize_t)data_len);

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
    bwa->ctx = ctx;
    VK_TRACE("create_buffer done bwa=%p", (void*)bwa);

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(bwa));
}

// ---------------------------------------------------------------------------
//  Create a persistently-mapped buffer.  Returns (buffer_handle, mapped_ptr)
//  so the caller can write directly into GPU-visible memory (e.g. from a
//  C-level surface generator) without an intermediate numpy copy.
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_create_mapped_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    Py_ssize_t size_bytes;
    static const char* kwlist[] = {"context", "size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist, &ctx_ptr, &size_bytes))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size_bytes;
    buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                      | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                     | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer vk_buffer;
    VmaAllocation allocation;
    VmaAllocationInfo ai;
    VkResult result = vmaCreateBuffer(ctx->allocator, &buffer_info, &alloc_info,
                                      &vk_buffer, &allocation, &ai);
    if (result != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create mapped Vulkan buffer");
        return NULL;
    }

    BufferWithAllocator* bwa = new BufferWithAllocator();
    bwa->base.buffer = vk_buffer;
    bwa->base.allocation = allocation;
    bwa->base.size = static_cast<VkDeviceSize>(size_bytes);
    bwa->allocator = ctx->allocator;
    bwa->ctx = ctx;

    return Py_BuildValue("nn",
        reinterpret_cast<Py_ssize_t>(bwa),
        reinterpret_cast<Py_ssize_t>(ai.pMappedData));
}

extern "C"
PyObject* vulkan_update_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t buf_ptr;
    Py_buffer py_buf;
    static const char* kwlist[] = {"buffer", "data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ny*"), (char**)kwlist, &buf_ptr, &py_buf))
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
    Py_ssize_t buf_ptr;
    static const char* kwlist[] = {"buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &buf_ptr))
        return NULL;

    BufferWithAllocator* bwa = reinterpret_cast<BufferWithAllocator*>(buf_ptr);

    // Queue for destruction once any in-flight frame that could reference
    // this buffer has retired. deferred_delete_slot routes to the slot
    // currently in flight (or being recorded), so the next reuse of that
    // slot's fence is what gates the actual vkDestroyBuffer.
    bwa->ctx->pending_deletes[deferred_delete_slot(bwa->ctx)].buffers.push_back(bwa);

    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_get_buffer_device_memory(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t buf_ptr;
    static const char* kwlist[] = {"buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &buf_ptr))
        return NULL;

    if (!buf_ptr) {
        PyErr_SetString(PyExc_ValueError, "buffer handle is NULL");
        return NULL;
    }
    BufferWithAllocator* bwa = reinterpret_cast<BufferWithAllocator*>(buf_ptr);
    VmaAllocationInfo alloc_info;
    vmaGetAllocationInfo(bwa->allocator, bwa->base.allocation, &alloc_info);

    // Return (device_memory, offset) — VMA may sub-allocate from a larger
    // block, so the caller needs the offset to find the data.
    return Py_BuildValue("Kn",
        reinterpret_cast<unsigned long long>(alloc_info.deviceMemory),
        static_cast<Py_ssize_t>(alloc_info.offset));
}

// ---------------------------------------------------------------------------
//  Draw state binding
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_vertex_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr, buf_ptr;
    int index, offset;
    static const char* kwlist[] = {"context", "buffer", "index", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nnii"), (char**)kwlist,
            &ctx_ptr, &buf_ptr, &index, &offset))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanBuffer* buf = reinterpret_cast<VulkanBuffer*>(buf_ptr);

    VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;
    VkDeviceSize vk_offset = static_cast<VkDeviceSize>(offset);
    vkCmdBindVertexBuffers(cmd, static_cast<uint32_t>(index), 1,
                           &buf->buffer, &vk_offset);

    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_set_descriptor_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr, buf_ptr;
    int index, offset;
    static const char* kwlist[] = {"context", "buffer", "index", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nnii"), (char**)kwlist,
            &ctx_ptr, &buf_ptr, &index, &offset))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanBuffer* buf = reinterpret_cast<VulkanBuffer*>(buf_ptr);

    if (index < 0 || index >= static_cast<int>(MAX_UBO_BINDINGS)) {
        PyErr_Format(PyExc_ValueError,
                     "UBO index %d out of range [0, %u)", index, MAX_UBO_BINDINGS);
        return NULL;
    }

    ctx->bound_ubo_buffers[index] = buf;
    ctx->descriptor_dirty = true;
    VK_TRACE("set_descriptor_buffer index=%d buf=%p size=%zd",
             index, (void*)buf, (Py_ssize_t)buf->size);

    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_set_fragment_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr, tex_ptr;
    int index;
    static const char* kwlist[] = {"context", "texture", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nni"), (char**)kwlist,
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
    VK_TRACE("set_fragment_texture index=%d tex=%p", index, (void*)tex);

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

    // Allocate fresh descriptor sets out of the per-frame pool. We never
    // reuse a set within a single command buffer recording: vkCmdBindDescriptorSets
    // captures the descriptor set state, and updating that set later
    // would invalidate the recorded binding (Vulkan disallows it without
    // the UPDATE_AFTER_BIND extension). The pool is reset at the start
    // of every frame so allocations stay cheap.
    VkDescriptorSetLayout layouts[2] = {
        ctx->descriptor_set_layout_ubo,
        ctx->descriptor_set_layout_textures,
    };
    VkDescriptorSet sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    {
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = ctx->descriptor_pools[frame];
        alloc.descriptorSetCount = 2;
        alloc.pSetLayouts = layouts;
        VkResult r = vkAllocateDescriptorSets(ctx->device, &alloc, sets);
        if (r != VK_SUCCESS) {
            // Pool exhausted — caller will see no binding update this draw,
            // but at least the previously bound set stays valid.
            return;
        }
    }

    // Write descriptors for EVERY layout binding (set 0 + set 1) on every
    // flush. User-bound resources go into the slots that have them; the
    // remaining slots are filled with the per-type dummy resources
    // created in vulkan_dummies.cpp. Vulkan validation requires every
    // shader-statically-referenced binding to have a valid descriptor;
    // skipping the write because "the shader's runtime path doesn't
    // sample" is not enough -- the SPIR-V's descriptor declaration
    // counts as static use, even when a specialization-constant if-gate
    // compiles the actual access away.
    VkDescriptorBufferInfo buffer_infos[MAX_UBO_BINDINGS] = {};
    VkWriteDescriptorSet writes[MAX_UBO_BINDINGS + MAX_TEXTURE_BINDINGS] = {};
    uint32_t write_count = 0;

    for (uint32_t i = 0; i < MAX_UBO_BINDINGS; i++) {
        VulkanBuffer* buf = ctx->bound_ubo_buffers[i];
        if (buf) {
            buffer_infos[i].buffer = buf->buffer;
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = buf->size;
        } else if (set0_descriptor_type(i) == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            buffer_infos[i].buffer = ctx->dummy.ssbo_buffer;
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = VK_WHOLE_SIZE;
        } else {
            buffer_infos[i].buffer = ctx->dummy.ubo_buffer;
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = VK_WHOLE_SIZE;
        }

        VkWriteDescriptorSet& w = writes[write_count++];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = sets[0];
        w.dstBinding = i;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = set0_descriptor_type(i);
        w.pBufferInfo = &buffer_infos[i];
    }

    VkDescriptorImageInfo image_infos[MAX_TEXTURE_BINDINGS] = {};

    for (uint32_t i = 0; i < MAX_TEXTURE_BINDINGS; i++) {
        VulkanTexture* tex = ctx->bound_textures[i];
        if (tex) {
            image_infos[i].sampler = tex->sampler;
            image_infos[i].imageView = tex->image_view;
            image_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else {
            dummy_image_info_for_binding(ctx, i, image_infos[i]);
        }

        VkWriteDescriptorSet& w = writes[write_count++];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = sets[1];
        w.dstBinding = i;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &image_infos[i];
    }

    if (write_count > 0)
        vkUpdateDescriptorSets(ctx->device, write_count, writes, 0, nullptr);

    {
        // Summarize what's bound — useful for diagnosing missing bindings
        // for shadow / multishadow descriptors.
        char ubo_state[MAX_UBO_BINDINGS + 1];
        for (uint32_t i = 0; i < MAX_UBO_BINDINGS; ++i)
            ubo_state[i] = ctx->bound_ubo_buffers[i] ? '1' : '0';
        ubo_state[MAX_UBO_BINDINGS] = '\0';
        char tex_state[MAX_TEXTURE_BINDINGS + 1];
        for (uint32_t i = 0; i < MAX_TEXTURE_BINDINGS; ++i)
            tex_state[i] = ctx->bound_textures[i] ? '1' : '0';
        tex_state[MAX_TEXTURE_BINDINGS] = '\0';
        VK_TRACE("flush_descriptors set0=%s set1=%s", ubo_state, tex_state);
    }

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
    Py_ssize_t ctx_ptr, index_buf_ptr;
    int primitive_type, index_count, index_offset, instance_count;
    static const char* kwlist[] = {"context", "index_buffer", "primitive_type",
                                   "index_count", "index_offset", "instance_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nniiii"), (char**)kwlist,
            &ctx_ptr, &index_buf_ptr, &primitive_type,
            &index_count, &index_offset, &instance_count))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanBuffer* idx_buf = reinterpret_cast<VulkanBuffer*>(index_buf_ptr);
    VK_TRACE("draw_indexed prim=%d count=%d offset=%d inst=%d in_pass=%d",
             primitive_type, index_count, index_offset, instance_count, ctx->in_render_pass ? 1 : 0);

    if (!ctx->in_render_pass)
        Py_RETURN_NONE;

    VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;

    // Flush any pending descriptor set updates.
    flush_descriptors(ctx);

    // Bind the index buffer at offset 0 and use firstIndex on the draw.
    // This is equivalent to binding with a byte offset and firstIndex=0,
    // but avoids re-binding the index buffer per draw inside a multitexture
    // loop (some drivers misbehave with repeated index-buffer binds).
    vkCmdBindIndexBuffer(cmd, idx_buf->buffer, 0, VK_INDEX_TYPE_UINT32);

    // Issue the draw call.  Primitive topology is baked into the pipeline
    // state, so the primitive_type parameter is not used here.
    (void)primitive_type;
    vkCmdDrawIndexed(cmd,
                     static_cast<uint32_t>(index_count),
                     static_cast<uint32_t>(instance_count),
                     static_cast<uint32_t>(index_offset),  // firstIndex
                     0,   // vertexOffset
                     0);  // firstInstance

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Depth/stencil state
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_depth_stencil_state(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    int depth_write, depth_test, compare_func;
    static const char* kwlist[] = {"context", "depth_write", "depth_test",
                                   "compare_func", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("niii"), (char**)kwlist,
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
