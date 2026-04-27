/*
 * Dummy descriptor resources used to satisfy validation when the active
 * shader's pipeline layout declares bindings the runtime path doesn't
 * actually sample/read. Each shader variant statically references all
 * the samplers/buffers in its descriptor set layout, even when its
 * specialization-constant gates compile away the actual access. Vulkan
 * validation requires every such binding to have a valid descriptor
 * written for the bound pipeline; otherwise it fires
 * VUID-vkCmdDrawIndexed-None-08114.
 *
 * To avoid making the hot path conditional, flush_descriptors writes
 * descriptors for every layout binding -- the user-bound resource if
 * one was provided, otherwise the dummy created here. Dummies are 1x1
 * (or 1x1x1 / 1x1x6) images with the right viewType for each slot, plus
 * tiny UBO/SSBO buffers for set-0 bindings.
 */

#include <Python.h>
#include "vulkan_internal.h"
#include <cstring>

namespace {

// Helper that creates a 1x...x1 image, allocates memory, builds an image
// view, and transitions the image to SHADER_READ_ONLY_OPTIMAL via a
// one-shot command buffer.
bool make_dummy_image(VulkanContext* ctx,
                      VkImageType image_type,
                      VkImageViewType view_type,
                      VkFormat format,
                      VkImageAspectFlags aspect,
                      uint32_t array_layers,
                      VkImageCreateFlags create_flags,
                      VkImage* out_image, VmaAllocation* out_alloc, VkImageView* out_view) {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.flags = create_flags;
    info.imageType = image_type;
    info.format = format;
    info.extent = {1, 1, 1};
    info.mipLevels = 1;
    info.arrayLayers = array_layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;

    if (vmaCreateImage(ctx->allocator, &info, &ai,
                       out_image, out_alloc, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = *out_image;
    vi.viewType = view_type;
    vi.format = format;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = array_layers;

    if (vkCreateImageView(ctx->device, &vi, nullptr, out_view) != VK_SUCCESS) {
        vmaDestroyImage(ctx->allocator, *out_image, *out_alloc);
        *out_image = VK_NULL_HANDLE;
        *out_alloc = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

void transition_dummy_to_read_only(VulkanContext* ctx, VkImage image,
                                   VkImageAspectFlags aspect, uint32_t layers) {
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = ctx->command_pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx->device, &cai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    // transition_image_layout in vulkan_texture.cpp is single-layer only;
    // the cubemap path needs all 6 layers transitioned, so emit the
    // barrier directly here for full coverage.
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = aspect;
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = layers;
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx->graphics_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->graphics_queue);
    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
}

bool make_dummy_buffer(VulkanContext* ctx, VkBufferUsageFlags usage,
                       VkBuffer* out_buf, VmaAllocation* out_alloc) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 64;  // tiny -- shader doesn't actually read from these
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    return vmaCreateBuffer(ctx->allocator, &bi, &ai,
                           out_buf, out_alloc, nullptr) == VK_SUCCESS;
}

}  // namespace

void create_dummy_resources(VulkanContext* ctx) {
    auto& d = ctx->dummy;
    constexpr VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkFormat depth_fmt = VK_FORMAT_D32_SFLOAT;

    // 2D color
    if (make_dummy_image(ctx, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
                         color_fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0,
                         &d.tex2d_image, &d.tex2d_alloc, &d.tex2d_view)) {
        transition_dummy_to_read_only(ctx, d.tex2d_image, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
    // 3D color
    if (make_dummy_image(ctx, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,
                         color_fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0,
                         &d.tex3d_image, &d.tex3d_alloc, &d.tex3d_view)) {
        transition_dummy_to_read_only(ctx, d.tex3d_image, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
    // Cube color (6 layers, CUBE_COMPATIBLE flag)
    if (make_dummy_image(ctx, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE,
                         color_fmt, VK_IMAGE_ASPECT_COLOR_BIT, 6,
                         VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
                         &d.texcube_image, &d.texcube_alloc, &d.texcube_view)) {
        transition_dummy_to_read_only(ctx, d.texcube_image, VK_IMAGE_ASPECT_COLOR_BIT, 6);
    }
    // 1D color
    if (make_dummy_image(ctx, VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D,
                         color_fmt, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0,
                         &d.tex1d_image, &d.tex1d_alloc, &d.tex1d_view)) {
        transition_dummy_to_read_only(ctx, d.tex1d_image, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
    // Depth (sampler2DShadow). Needs DEPTH_STENCIL_ATTACHMENT_BIT? No --
    // SAMPLED_BIT is enough since we never render to it.
    if (make_dummy_image(ctx, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
                         depth_fmt, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 0,
                         &d.depth_image, &d.depth_alloc, &d.depth_view)) {
        transition_dummy_to_read_only(ctx, d.depth_image, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
    }

    // Color sampler (compareEnable=false).
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_NEAREST;
        si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vkCreateSampler(ctx->device, &si, nullptr, &d.color_sampler);
    }
    // Shadow sampler (compareEnable=true, matching sampler2DShadow).
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        si.compareEnable = VK_TRUE;
        si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        vkCreateSampler(ctx->device, &si, nullptr, &d.shadow_sampler);
    }

    // Set-0 UBO and SSBO buffers (single shared instance each).
    make_dummy_buffer(ctx, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      &d.ubo_buffer, &d.ubo_alloc);
    make_dummy_buffer(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      &d.ssbo_buffer, &d.ssbo_alloc);
}

void destroy_dummy_resources(VulkanContext* ctx) {
    auto& d = ctx->dummy;
    if (d.color_sampler != VK_NULL_HANDLE)
        vkDestroySampler(ctx->device, d.color_sampler, nullptr);
    if (d.shadow_sampler != VK_NULL_HANDLE)
        vkDestroySampler(ctx->device, d.shadow_sampler, nullptr);
    auto destroy_img = [&](VkImage img, VkImageView v, VmaAllocation a) {
        if (v != VK_NULL_HANDLE) vkDestroyImageView(ctx->device, v, nullptr);
        if (img != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator, img, a);
    };
    destroy_img(d.tex2d_image, d.tex2d_view, d.tex2d_alloc);
    destroy_img(d.tex3d_image, d.tex3d_view, d.tex3d_alloc);
    destroy_img(d.texcube_image, d.texcube_view, d.texcube_alloc);
    destroy_img(d.tex1d_image, d.tex1d_view, d.tex1d_alloc);
    destroy_img(d.depth_image, d.depth_view, d.depth_alloc);
    if (d.ubo_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx->allocator, d.ubo_buffer, d.ubo_alloc);
    if (d.ssbo_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(ctx->allocator, d.ssbo_buffer, d.ssbo_alloc);
    d = {};
}

void dummy_image_info_for_binding(VulkanContext* ctx, uint32_t binding,
                                  VkDescriptorImageInfo& info) {
    auto& d = ctx->dummy;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Per-binding mapping mirrors the mesh / impostor / raycast shader
    // declarations in src/bundles/graphics/src/shaders/. Update if a
    // shader adds a sampler at a new slot or changes a slot's type.
    switch (binding) {
        case 1:  // sampler3D tex3d
        case 5:  // sampler3D ambient_tex3d
            info.sampler = d.color_sampler;
            info.imageView = d.tex3d_view;
            break;
        case 2:  // samplerCube texcube
            info.sampler = d.color_sampler;
            info.imageView = d.texcube_view;
            break;
        case 3:  // sampler2DShadow shadow_map
        case 4:  // sampler2DShadow multishadow_map
            info.sampler = d.shadow_sampler;
            info.imageView = d.depth_view;
            break;
        case 6:  // sampler1D colormap_tex
            info.sampler = d.color_sampler;
            info.imageView = d.tex1d_view;
            break;
        case 0:  // sampler2D tex2d
        case 7:  // (reserved 2D slot)
        default:
            info.sampler = d.color_sampler;
            info.imageView = d.tex2d_view;
            break;
    }
}
