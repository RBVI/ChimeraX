/*
 * Vulkan texture management: create, destroy, and bind GPU textures.
 */

#include <Python.h>
#include "vulkan_internal.h"

// ---------------------------------------------------------------------------
//  Extended texture struct that carries the device and allocator handles
//  so that destroy can clean up without needing the context.
//  The base VulkanTexture sits at offset 0, so a pointer to this struct
//  can be safely reinterpret_cast to VulkanTexture*.
// ---------------------------------------------------------------------------
struct TextureWithDevice : VulkanTexture {
    VkDevice     device;
    VmaAllocator allocator;
};

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
    long ctx_ptr;
    int width, height;
    int format;  // 0=RGBA8, 1=R8, 2=Depth32Float, 3=Depth32Float_Stencil8, 4=R32Float
    int linear_filter;
    int depth_compare;

    static const char* kwlist[] = {"context", "width", "height", "format",
                                   "linear_filter", "depth_compare", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("liiiii"), (char**)kwlist,
            &ctx_ptr, &width, &height, &format, &linear_filter, &depth_compare))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);

    // Map format enum to Vulkan format.
    VkFormat vk_format;
    switch (format) {
        case 0: vk_format = VK_FORMAT_R8G8B8A8_UNORM; break;
        case 1: vk_format = VK_FORMAT_R8_UNORM; break;
        case 2: vk_format = VK_FORMAT_D32_SFLOAT; break;
        case 3: vk_format = VK_FORMAT_D32_SFLOAT_S8_UINT; break;
        case 4: vk_format = VK_FORMAT_R32_SFLOAT; break;
        default: vk_format = VK_FORMAT_R8G8B8A8_UNORM; break;
    }

    bool is_depth = (format == 2 || format == 3);

    // Determine image usage flags.
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    if (is_depth) {
        usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    } else {
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    }

    // Create VkImage via VMA.
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = vk_format;
    image_info.extent = {static_cast<uint32_t>(width),
                         static_cast<uint32_t>(height), 1};
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
    VkResult result = vmaCreateImage(ctx->allocator, &image_info, &alloc_info,
                                     &image, &allocation, nullptr);
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
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = vk_format;
    view_info.subresourceRange.aspectMask = aspect;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView image_view;
    result = vkCreateImageView(ctx->device, &view_info, nullptr, &image_view);
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
    result = vkCreateSampler(ctx->device, &sampler_info, nullptr, &sampler);
    if (result != VK_SUCCESS) {
        vkDestroyImageView(ctx->device, image_view, nullptr);
        vmaDestroyImage(ctx->allocator, image, allocation);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Vulkan sampler");
        return NULL;
    }

    // Transition image layout to shader-read-only.
    VkCommandBuffer cmd = begin_one_shot(ctx);
    transition_image_layout(cmd, image,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            aspect);
    end_one_shot(ctx, cmd);

    // Allocate and fill the texture struct.
    TextureWithDevice* tex = new TextureWithDevice();
    tex->image = image;
    tex->allocation = allocation;
    tex->image_view = image_view;
    tex->sampler = sampler;
    tex->format = vk_format;
    tex->width = static_cast<uint32_t>(width);
    tex->height = static_cast<uint32_t>(height);
    tex->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    tex->device = ctx->device;
    tex->allocator = ctx->allocator;

    return PyLong_FromLong(reinterpret_cast<long>(tex));
}

// ---------------------------------------------------------------------------
//  Destroy texture
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_destroy_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    long tex_ptr;
    static const char* kwlist[] = {"texture", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("l"), (char**)kwlist, &tex_ptr))
        return NULL;

    TextureWithDevice* tex = reinterpret_cast<TextureWithDevice*>(tex_ptr);

    vkDestroySampler(tex->device, tex->sampler, nullptr);
    vkDestroyImageView(tex->device, tex->image_view, nullptr);
    vmaDestroyImage(tex->allocator, tex->image, tex->allocation);
    delete tex;

    Py_RETURN_NONE;
}
