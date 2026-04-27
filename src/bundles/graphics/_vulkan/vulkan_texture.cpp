/*
 * Vulkan texture management: create, destroy, and bind GPU textures.
 */

#include <Python.h>
#include "vulkan_internal.h"

// TextureWithDevice is declared in vulkan_internal.h so the
// deferred-deletion queue can free textures without a back-reference here.

// ---------------------------------------------------------------------------
//  One-shot command buffer helpers for layout transitions.
// ---------------------------------------------------------------------------

static VkCommandBuffer begin_one_shot(VulkanContext* ctx) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = ctx->command_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx->device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

static void end_one_shot(VulkanContext* ctx, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx->graphics_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->graphics_queue);
    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
}

// ---------------------------------------------------------------------------
//  Create texture
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_create_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    int width, height;
    // format codes:
    //   0=RGBA8 1=R8 2=Depth32Float 3=Depth32+Stencil 4=R32Float
    //   5=R16_UNORM 6=R16_SFLOAT
    int format;
    int linear_filter;
    int depth_compare;
    int dimension = 2;     // 1, 2, or 3
    int depth = 1;         // for 3D textures
    int storage = 0;       // 1 = also usable as a compute-shader storage image

    static const char* kwlist[] = {"context", "width", "height", "format",
                                   "linear_filter", "depth_compare",
                                   "dimension", "depth", "storage", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("niiiii|iii"), (char**)kwlist,
            &ctx_ptr, &width, &height, &format, &linear_filter, &depth_compare,
            &dimension, &depth, &storage))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("create_texture dim=%d %dx%dx%d fmt=%d linear=%d depth_cmp=%d storage=%d",
             dimension, width, height, depth, format, linear_filter, depth_compare, storage);

    // Map format enum to Vulkan format.
    VkFormat vk_format;
    switch (format) {
        case 0: vk_format = VK_FORMAT_R8G8B8A8_UNORM; break;
        case 1: vk_format = VK_FORMAT_R8_UNORM; break;
        case 2: vk_format = VK_FORMAT_D32_SFLOAT; break;
        case 3: vk_format = VK_FORMAT_D32_SFLOAT_S8_UINT; break;
        case 4: vk_format = VK_FORMAT_R32_SFLOAT; break;
        case 5: vk_format = VK_FORMAT_R16_UNORM; break;
        case 6: vk_format = VK_FORMAT_R16_SFLOAT; break;
        case 7: vk_format = VK_FORMAT_R16_SNORM; break;
        case 8: vk_format = VK_FORMAT_R8_SNORM;  break;
        default: vk_format = VK_FORMAT_R8G8B8A8_UNORM; break;
    }

    bool is_depth = (format == 2 || format == 3);

    // Image type/view depend on dimension.
    VkImageType image_type;
    VkImageViewType view_type;
    switch (dimension) {
        case 1: image_type = VK_IMAGE_TYPE_1D;
                view_type  = VK_IMAGE_VIEW_TYPE_1D; break;
        case 3: image_type = VK_IMAGE_TYPE_3D;
                view_type  = VK_IMAGE_VIEW_TYPE_3D; break;
        default: image_type = VK_IMAGE_TYPE_2D;
                 view_type  = VK_IMAGE_VIEW_TYPE_2D; break;
    }

    // Determine image usage flags.
    // SAMPLED_BIT: required for all textures we sample from.
    // TRANSFER_DST_BIT: needed so vulkan_update_texture can copy data in.
    // For RGBA8 2D color textures we additionally allow render-target use,
    // since some callers reuse the path for off-screen render targets.
    // We avoid COLOR_ATTACHMENT_BIT for single-channel formats: R16_SNORM /
    // R8_SNORM aren't required to support color attachment, and asking for
    // it can flip the driver to a different (non-sampleable on some sizes)
    // memory layout that quietly breaks NPOT sampling.
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT
                            | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (is_depth) {
        usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    } else if (dimension == 2 && format == 0) {
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    // Compute-shader writeable target. Caller is responsible for picking a
    // format that supports STORAGE_IMAGE — RGBA8_UNORM (0) and R32_SFLOAT
    // (4) are the two we use, both of which are core-required.
    if (storage && !is_depth) {
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    // Create VkImage via VMA.
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = image_type;
    image_info.format = vk_format;
    image_info.extent = {static_cast<uint32_t>(width),
                         static_cast<uint32_t>(height),
                         static_cast<uint32_t>(depth)};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

    VkImage image;
    VmaAllocation allocation;
    VK_TRACE("  vmaCreateImage usage=0x%x", (unsigned)usage);
    VkResult result = vmaCreateImage(ctx->allocator, &image_info, &alloc_info,
                                     &image, &allocation, nullptr);
    VK_TRACE("  vmaCreateImage -> %d image=%p alloc=%p", (int)result, (void*)image, (void*)allocation);
    if (result != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Vulkan image");
        return NULL;
    }

    // Create image view.
    VkImageAspectFlags aspect = is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT
                                         : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = view_type;
    view_info.format = vk_format;
    view_info.subresourceRange.aspectMask = aspect;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    // Broadcast the R channel of single-channel (luminance) color formats
    // to G and B so a sampler reading .rgba sees (L, L, L, 1). Mirrors the
    // OpenGL backend's GL_TEXTURE_SWIZZLE_G/B = GL_RED setup that volume
    // colormap textures rely on; data textures the shader reads with .r
    // are unaffected. Depth formats are excluded.
    //
    // Storage images are also excluded: VUID-VkWriteDescriptorSet-
    // descriptorType-00336 requires that any image view bound as
    // VK_DESCRIPTOR_TYPE_STORAGE_IMAGE be created with the identity
    // swizzle. Compute kernels that bind these as `image2D` only ever
    // touch .r, so the swizzle was useless to them anyway.
    bool single_channel_color = (format == 1 || format == 4 || format == 5 ||
                                 format == 6 || format == 7 || format == 8);
    if (single_channel_color && !storage) {
        view_info.components.g = VK_COMPONENT_SWIZZLE_R;
        view_info.components.b = VK_COMPONENT_SWIZZLE_R;
    }

    VkImageView image_view;
    VK_TRACE("  vkCreateImageView");
    result = vkCreateImageView(ctx->device, &view_info, nullptr, &image_view);
    VK_TRACE("  vkCreateImageView -> %d view=%p", (int)result, (void*)image_view);
    if (result != VK_SUCCESS) {
        vmaDestroyImage(ctx->allocator, image, allocation);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Vulkan image view");
        return NULL;
    }

    // Create sampler.
    VkFilter filter = linear_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = filter;
    sampler_info.minFilter = filter;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.maxLod = 0.0f;

    if (depth_compare) {
        sampler_info.compareEnable = VK_TRUE;
        sampler_info.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    }

    VkSampler sampler;
    VK_TRACE("  vkCreateSampler");
    result = vkCreateSampler(ctx->device, &sampler_info, nullptr, &sampler);
    VK_TRACE("  vkCreateSampler -> %d sampler=%p", (int)result, (void*)sampler);
    if (result != VK_SUCCESS) {
        vkDestroyImageView(ctx->device, image_view, nullptr);
        vmaDestroyImage(ctx->allocator, image, allocation);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Vulkan sampler");
        return NULL;
    }

    // Transition image layout to shader-read-only.
    VK_TRACE("  begin_one_shot");
    VkCommandBuffer cmd = begin_one_shot(ctx);
    VK_TRACE("  transition UNDEFINED->SHADER_READ_ONLY aspect=0x%x", (unsigned)aspect);
    transition_image_layout(cmd, image,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            aspect);
    VK_TRACE("  end_one_shot");
    end_one_shot(ctx, cmd);
    VK_TRACE("  create_texture done");

    // Allocate and fill the texture struct.
    TextureWithDevice* tex = new TextureWithDevice();
    tex->image = image;
    tex->allocation = allocation;
    tex->image_view = image_view;
    tex->sampler = sampler;
    tex->format = vk_format;
    tex->width = static_cast<uint32_t>(width);
    tex->height = static_cast<uint32_t>(height);
    tex->depth = static_cast<uint32_t>(depth);
    tex->dimension = static_cast<uint32_t>(dimension);
    tex->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    tex->device = ctx->device;
    tex->allocator = ctx->allocator;
    tex->ctx = ctx;

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(tex));
}

// ---------------------------------------------------------------------------
//  Upload pixel data into an existing texture via a staging buffer.
//  Caller passes a Python buffer-protocol object (numpy array) sized to
//  match the texture's full extent in the texture's pixel format.
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_update_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t tex_ptr;
    Py_buffer py_buf;
    static const char* kwlist[] = {"texture", "data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ny*"), (char**)kwlist, &tex_ptr, &py_buf))
        return NULL;

    TextureWithDevice* tex = reinterpret_cast<TextureWithDevice*>(tex_ptr);
    VulkanContext* ctx = tex->ctx;
    Py_ssize_t data_len = py_buf.len;
    VK_TRACE("update_texture tex=%p size=%zd %ux%ux%u",
             (void*)tex, (Py_ssize_t)data_len,
             tex->width, tex->height, tex->depth);

    // Allocate a host-visible staging buffer.
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = static_cast<VkDeviceSize>(data_len);
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
             | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo staging_info{};
    if (vmaCreateBuffer(ctx->allocator, &bi, &ai,
                        &staging, &staging_alloc, &staging_info) != VK_SUCCESS) {
        PyBuffer_Release(&py_buf);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create texture staging buffer");
        return NULL;
    }

    // Copy CPU data into the staging buffer's mapped memory.
    if (staging_info.pMappedData) {
        memcpy(staging_info.pMappedData, py_buf.buf, data_len);
    } else {
        void* mapped = nullptr;
        vmaMapMemory(ctx->allocator, staging_alloc, &mapped);
        memcpy(mapped, py_buf.buf, data_len);
        vmaUnmapMemory(ctx->allocator, staging_alloc);
    }
    PyBuffer_Release(&py_buf);

    // One-shot command buffer: transition to TRANSFER_DST, copy, transition
    // to SHADER_READ_ONLY.
    VkCommandBuffer cmd = begin_one_shot(ctx);

    VkImageLayout old_layout = tex->current_layout != VK_IMAGE_LAYOUT_UNDEFINED
        ? tex->current_layout : VK_IMAGE_LAYOUT_UNDEFINED;
    transition_image_layout(cmd, tex->image, old_layout,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_ASPECT_COLOR_BIT);

    VkBufferImageCopy copy{};
    copy.bufferOffset = 0;
    copy.bufferRowLength = 0;     // tightly packed
    copy.bufferImageHeight = 0;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = {0, 0, 0};
    copy.imageExtent = {tex->width, tex->height, tex->depth};
    vkCmdCopyBufferToImage(cmd, staging, tex->image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &copy);

    transition_image_layout(cmd, tex->image,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_ASPECT_COLOR_BIT);
    tex->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    end_one_shot(ctx, cmd);

    // Staging is finished now (end_one_shot waits for queue idle).
    vmaDestroyBuffer(ctx->allocator, staging, staging_alloc);

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Destroy texture
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_destroy_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t tex_ptr;
    static const char* kwlist[] = {"texture", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &tex_ptr))
        return NULL;

    TextureWithDevice* tex = reinterpret_cast<TextureWithDevice*>(tex_ptr);
    VK_TRACE("destroy_texture %p", (void*)tex);

    // Defer destruction until any in-flight frame that could reference
    // this texture has finished. deferred_delete_slot picks the right
    // queue depending on whether we're inside or outside frame recording.
    tex->ctx->pending_deletes[deferred_delete_slot(tex->ctx)].textures.push_back(tex);

    Py_RETURN_NONE;
}
