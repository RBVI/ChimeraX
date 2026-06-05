/*
 * OIT (Weighted Blended Order-Independent Transparency) for Vulkan.
 *
 * Creates a separate render pass with two color attachments:
 *   0: accum  (RGBA16F) — additive weighted color + alpha accumulation
 *   1: reveal (R8)      — multiplicative transmittance product
 * The depth buffer is shared with the main render pass (read-only).
 *
 * Usage:
 *   create_oit_resources  — allocate images, render pass, framebuffer
 *   begin_oit_pass        — end main pass, begin OIT pass (clears)
 *   <draw transparent>    — using OIT pipeline (oit_mode=1)
 *   end_oit_pass          — end OIT pass, resume main pass
 *   <composite quad>      — fullscreen quad reads accum+reveal textures
 *   destroy_oit_resources — cleanup
 */

#include <Python.h>
#include "vulkan_internal.h"

// Helpers from vulkan_framebuffer.cpp.
namespace {

constexpr VkFormat kAccumFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kRevealFormat = VK_FORMAT_R8_UNORM;
constexpr VkFormat kDepthFormat  = VK_FORMAT_D32_SFLOAT;

bool create_oit_image(VulkanContext* ctx, uint32_t w, uint32_t h,
                      VkFormat format, VkImageUsageFlags usage,
                      VkImageAspectFlags aspect,
                      VkImage* image, VmaAllocation* alloc, VkImageView* view)
{
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {w, h, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;

    if (vmaCreateImage(ctx->allocator, &info, &ai, image, alloc, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo vinfo{};
    vinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vinfo.image = *image;
    vinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vinfo.format = format;
    vinfo.subresourceRange.aspectMask = aspect;
    vinfo.subresourceRange.baseMipLevel = 0;
    vinfo.subresourceRange.levelCount = 1;
    vinfo.subresourceRange.baseArrayLayer = 0;
    vinfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(ctx->device, &vinfo, nullptr, view) != VK_SUCCESS) {
        vmaDestroyImage(ctx->allocator, *image, *alloc);
        *image = VK_NULL_HANDLE;
        *alloc = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

VkRenderPass create_oit_render_pass(VkDevice device) {
    VkAttachmentDescription attachments[3] = {};

    // 0: accum (RGBA16F, clear to 0, store for composite read)
    attachments[0].format = kAccumFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // 1: revealage (R8, clear to 1, store for composite read)
    attachments[1].format = kRevealFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // 2: depth (D32, load from opaque pass, read-only)
    attachments[2].format = kDepthFormat;
    attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_refs[2] = {};
    color_refs[0].attachment = 0;
    color_refs[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_refs[1].attachment = 1;
    color_refs[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_ref{};
    depth_ref.attachment = 2;
    depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 2;
    subpass.pColorAttachments = color_refs;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 3;
    rp.pAttachments = attachments;
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;
    rp.dependencyCount = 1;
    rp.pDependencies = &dep;

    VkRenderPass pass = VK_NULL_HANDLE;
    vkCreateRenderPass(device, &rp, nullptr, &pass);
    return pass;
}

void destroy_oit(VulkanContext* ctx) {
    if (ctx->oit_framebuffer) vkDestroyFramebuffer(ctx->device, ctx->oit_framebuffer, nullptr);
    if (ctx->oit_render_pass) vkDestroyRenderPass(ctx->device, ctx->oit_render_pass, nullptr);
    if (ctx->oit_accum_view) vkDestroyImageView(ctx->device, ctx->oit_accum_view, nullptr);
    if (ctx->oit_accum_image) vmaDestroyImage(ctx->allocator, ctx->oit_accum_image, ctx->oit_accum_alloc);
    if (ctx->oit_reveal_view) vkDestroyImageView(ctx->device, ctx->oit_reveal_view, nullptr);
    if (ctx->oit_reveal_image) vmaDestroyImage(ctx->allocator, ctx->oit_reveal_image, ctx->oit_reveal_alloc);

    // Clean up samplable texture wrappers (don't destroy underlying
    // images — already handled above).
    if (ctx->oit_accum_texture.sampler)
        vkDestroySampler(ctx->device, ctx->oit_accum_texture.sampler, nullptr);
    if (ctx->oit_reveal_texture.sampler)
        vkDestroySampler(ctx->device, ctx->oit_reveal_texture.sampler, nullptr);

    ctx->oit_framebuffer = VK_NULL_HANDLE;
    ctx->oit_render_pass = VK_NULL_HANDLE;
    ctx->oit_accum_image = VK_NULL_HANDLE;
    ctx->oit_accum_alloc = VK_NULL_HANDLE;
    ctx->oit_accum_view = VK_NULL_HANDLE;
    ctx->oit_reveal_image = VK_NULL_HANDLE;
    ctx->oit_reveal_alloc = VK_NULL_HANDLE;
    ctx->oit_reveal_view = VK_NULL_HANDLE;
    ctx->oit_accum_texture = {};
    ctx->oit_reveal_texture = {};
    ctx->oit_width = 0;
    ctx->oit_height = 0;
}

}  // namespace

// ---------------------------------------------------------------------------
//  Public API
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_create_oit_resources(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    int width, height;
    static const char* kwlist[] = {"context", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nii"), (char**)kwlist, &ctx_ptr, &width, &height))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);

    // Tear down old resources if size changed.
    if (ctx->oit_render_pass && (ctx->oit_width != (uint32_t)width ||
                                  ctx->oit_height != (uint32_t)height)) {
        vkDeviceWaitIdle(ctx->device);
        destroy_oit(ctx);
    }
    if (ctx->oit_render_pass) Py_RETURN_NONE;  // Already created at this size.

    ctx->oit_width = (uint32_t)width;
    ctx->oit_height = (uint32_t)height;

    // Create images.
    if (!create_oit_image(ctx, width, height, kAccumFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            &ctx->oit_accum_image, &ctx->oit_accum_alloc, &ctx->oit_accum_view)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create OIT accum image");
        return NULL;
    }
    if (!create_oit_image(ctx, width, height, kRevealFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            &ctx->oit_reveal_image, &ctx->oit_reveal_alloc, &ctx->oit_reveal_view)) {
        destroy_oit(ctx);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create OIT revealage image");
        return NULL;
    }

    // Create render pass.
    ctx->oit_render_pass = create_oit_render_pass(ctx->device);
    if (!ctx->oit_render_pass) {
        destroy_oit(ctx);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create OIT render pass");
        return NULL;
    }

    // Create framebuffer (accum + reveal + main depth).
    VkImageView views[3] = {
        ctx->oit_accum_view,
        ctx->oit_reveal_view,
        ctx->depth_image.image_view
    };
    VkFramebufferCreateInfo fbi{};
    fbi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbi.renderPass = ctx->oit_render_pass;
    fbi.attachmentCount = 3;
    fbi.pAttachments = views;
    fbi.width = width;
    fbi.height = height;
    fbi.layers = 1;
    if (vkCreateFramebuffer(ctx->device, &fbi, nullptr, &ctx->oit_framebuffer) != VK_SUCCESS) {
        destroy_oit(ctx);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create OIT framebuffer");
        return NULL;
    }

    // Create nearest-neighbor sampler for the composite pass.
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    ctx->oit_accum_texture.image = ctx->oit_accum_image;
    ctx->oit_accum_texture.image_view = ctx->oit_accum_view;
    ctx->oit_accum_texture.width = width;
    ctx->oit_accum_texture.height = height;
    vkCreateSampler(ctx->device, &si, nullptr, &ctx->oit_accum_texture.sampler);

    ctx->oit_reveal_texture.image = ctx->oit_reveal_image;
    ctx->oit_reveal_texture.image_view = ctx->oit_reveal_view;
    ctx->oit_reveal_texture.width = width;
    ctx->oit_reveal_texture.height = height;
    vkCreateSampler(ctx->device, &si, nullptr, &ctx->oit_reveal_texture.sampler);

    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_destroy_oit_resources(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    static const char* kwlist[] = {"context", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &ctx_ptr))
        return NULL;
    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    vkDeviceWaitIdle(ctx->device);
    destroy_oit(ctx);
    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_begin_oit_pass(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    static const char* kwlist[] = {"context", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &ctx_ptr))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    if (!ctx->in_frame || !ctx->oit_render_pass) Py_RETURN_NONE;

    FrameData& frame = ctx->frames[ctx->current_frame];

    // End the current main render pass so the depth buffer is available.
    if (ctx->in_render_pass) {
        vkCmdEndRenderPass(frame.command_buffer);
        ctx->in_render_pass = false;
        ctx->render_pass_stack.clear();
    }

    // Begin the OIT render pass.
    VkClearValue clear_values[3] = {};
    clear_values[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};  // accum
    clear_values[1].color = {{1.0f, 0.0f, 0.0f, 0.0f}};  // revealage = 1.0
    clear_values[2].depthStencil = {1.0f, 0};              // not cleared (LOAD)

    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = ctx->oit_render_pass;
    rp.framebuffer = ctx->oit_framebuffer;
    rp.renderArea.offset = {0, 0};
    rp.renderArea.extent = {ctx->oit_width, ctx->oit_height};
    rp.clearValueCount = 3;
    rp.pClearValues = clear_values;

    vkCmdBeginRenderPass(frame.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);
    ctx->in_render_pass = true;

    // Re-apply viewport/scissor on the new render pass.
    if (ctx->has_viewport) {
        vkCmdSetViewport(frame.command_buffer, 0, 1, &ctx->pending_viewport);
        vkCmdSetScissor(frame.command_buffer, 0, 1, &ctx->pending_scissor);
    }

    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_end_oit_pass(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    static const char* kwlist[] = {"context", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &ctx_ptr))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    if (!ctx->in_frame) Py_RETURN_NONE;

    FrameData& frame = ctx->frames[ctx->current_frame];

    // End the OIT render pass.
    if (ctx->in_render_pass) {
        vkCmdEndRenderPass(frame.command_buffer);
        ctx->in_render_pass = false;
    }

    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_get_oit_textures(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    static const char* kwlist[] = {"context", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &ctx_ptr))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    return Py_BuildValue("nn",
        reinterpret_cast<Py_ssize_t>(&ctx->oit_accum_texture),
        reinterpret_cast<Py_ssize_t>(&ctx->oit_reveal_texture));
}
