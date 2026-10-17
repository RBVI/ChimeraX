/*
 * Offscreen framebuffer support for Vulkan image capture.
 *
 * Backs the Python VulkanFramebuffer class. While the renderer's
 * push_framebuffer makes one of these active, the swapchain path is
 * bypassed: begin_frame skips vkAcquireNextImage, begin_render_pass
 * targets the offscreen framebuffer/render-pass, and end_frame skips
 * vkQueuePresentKHR (just submits + waits on the fence).
 *
 * After rendering, vulkan_read_framebuffer copies the color image into a
 * host-visible staging buffer and memcpys the contents into a Python
 * buffer (numpy array). Output layout is RGBA8 row-major, top-down --
 * caller flips Y if needed.
 */

#include <Python.h>
#include "vulkan_internal.h"
#include <cstring>

namespace {

constexpr VkFormat kColorFormat8 = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kColorFormat16 = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// Build the render pass that the offscreen framebuffer's draws use.
// The color attachment ends in TRANSFER_SRC_OPTIMAL so the readback copy
// can happen without an extra layout transition.
VkRenderPass create_offscreen_render_pass(VkDevice device, VkFormat color_format) {
    VkAttachmentDescription attachments[2] = {};

    attachments[0].format = color_format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    attachments[1].format = kDepthFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Depth must survive the pass: later passes in the same capture (OIT
    // transparency, overlay resume) load it back as an attachment.
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_ref{};
    depth_ref.attachment = 1;
    depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 2;
    rp.pAttachments = attachments;
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;
    rp.dependencyCount = 1;
    rp.pDependencies = &dep;

    VkRenderPass render_pass = VK_NULL_HANDLE;
    vkCreateRenderPass(device, &rp, nullptr, &render_pass);
    return render_pass;
}

bool create_image(VulkanContext* ctx, uint32_t w, uint32_t h, VkFormat format,
                  VkImageUsageFlags usage, VkImageAspectFlags aspect,
                  VkImage* out_image, VmaAllocation* out_alloc, VkImageView* out_view) {
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

    if (vmaCreateImage(ctx->allocator, &info, &ai, out_image, out_alloc, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo vinfo{};
    vinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vinfo.image = *out_image;
    vinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vinfo.format = format;
    vinfo.subresourceRange.aspectMask = aspect;
    vinfo.subresourceRange.baseMipLevel = 0;
    vinfo.subresourceRange.levelCount = 1;
    vinfo.subresourceRange.baseArrayLayer = 0;
    vinfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(ctx->device, &vinfo, nullptr, out_view) != VK_SUCCESS) {
        vmaDestroyImage(ctx->allocator, *out_image, *out_alloc);
        *out_image = VK_NULL_HANDLE;
        *out_alloc = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

void destroy_offscreen(VulkanContext* ctx, OffscreenFramebuffer* fb) {
    if (!fb) return;
    if (fb->framebuffer != VK_NULL_HANDLE)
        vkDestroyFramebuffer(ctx->device, fb->framebuffer, nullptr);
    if (fb->render_pass != VK_NULL_HANDLE)
        vkDestroyRenderPass(ctx->device, fb->render_pass, nullptr);
    if (fb->color_view != VK_NULL_HANDLE)
        vkDestroyImageView(ctx->device, fb->color_view, nullptr);
    if (fb->color_image != VK_NULL_HANDLE)
        vmaDestroyImage(ctx->allocator, fb->color_image, fb->color_alloc);
    if (fb->depth_view != VK_NULL_HANDLE)
        vkDestroyImageView(ctx->device, fb->depth_view, nullptr);
    if (fb->depth_image != VK_NULL_HANDLE)
        vmaDestroyImage(ctx->allocator, fb->depth_image, fb->depth_alloc);
    if (fb->color_texture.sampler != VK_NULL_HANDLE)
        vkDestroySampler(ctx->device, fb->color_texture.sampler, nullptr);
    delete fb;
}

}  // namespace

void destroy_offscreen_resource_internal(VulkanContext* ctx,
                                         OffscreenFramebuffer* framebuffer)
{
    destroy_offscreen(ctx, framebuffer);
}

extern "C"
PyObject* vulkan_create_offscreen_framebuffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    int width, height;
    int alpha = 0;
    int color_bits = 8;
    static const char* kwlist[] = {"context", "width", "height", "alpha",
                                   "color_bits", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nii|ii"), (char**)kwlist,
            &ctx_ptr, &width, &height, &alpha, &color_bits))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("create_offscreen_framebuffer %dx%d alpha=%d", width, height, alpha);

    if (width <= 0 || height <= 0) {
        PyErr_SetString(PyExc_ValueError, "Framebuffer size must be positive");
        return NULL;
    }

    OffscreenFramebuffer* fb = new OffscreenFramebuffer{};
    fb->width = (uint32_t)width;
    fb->height = (uint32_t)height;
    fb->color_format = color_bits == 16 ? kColorFormat16 : kColorFormat8;
    fb->alpha = (alpha != 0);
    fb->color_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    fb->color_texture = {};

    if (!create_image(ctx, fb->width, fb->height, fb->color_format,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT,
                      &fb->color_image, &fb->color_alloc, &fb->color_view)) {
        delete fb;
        PyErr_SetString(PyExc_RuntimeError, "Failed to create offscreen color image");
        return NULL;
    }

    if (!create_image(ctx, fb->width, fb->height, kDepthFormat,
                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      VK_IMAGE_ASPECT_DEPTH_BIT,
                      &fb->depth_image, &fb->depth_alloc, &fb->depth_view)) {
        destroy_offscreen(ctx, fb);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create offscreen depth image");
        return NULL;
    }

    fb->render_pass = create_offscreen_render_pass(ctx->device, fb->color_format);
    if (fb->render_pass == VK_NULL_HANDLE) {
        destroy_offscreen(ctx, fb);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create offscreen render pass");
        return NULL;
    }

    VkImageView attachments[2] = { fb->color_view, fb->depth_view };
    VkFramebufferCreateInfo fbi{};
    fbi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbi.renderPass = fb->render_pass;
    fbi.attachmentCount = 2;
    fbi.pAttachments = attachments;
    fbi.width = fb->width;
    fbi.height = fb->height;
    fbi.layers = 1;
    if (vkCreateFramebuffer(ctx->device, &fbi, nullptr, &fb->framebuffer) != VK_SUCCESS) {
        destroy_offscreen(ctx, fb);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create offscreen framebuffer");
        return NULL;
    }

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.maxLod = 1.0f;
    if (vkCreateSampler(ctx->device, &sampler_info, nullptr,
                        &fb->color_texture.sampler) != VK_SUCCESS) {
        destroy_offscreen(ctx, fb);
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to create offscreen color sampler");
        return NULL;
    }
    fb->color_texture.image = fb->color_image;
    fb->color_texture.image_view = fb->color_view;
    fb->color_texture.allocation = VK_NULL_HANDLE;
    fb->color_texture.format = fb->color_format;
    fb->color_texture.width = fb->width;
    fb->color_texture.height = fb->height;
    fb->color_texture.depth = 1;
    fb->color_texture.dimension = 2;
    fb->color_texture.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    ctx->live_offscreens.insert(fb);
    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(fb));
}

extern "C"
PyObject* vulkan_get_offscreen_texture(PyObject* self, PyObject* args,
                                        PyObject* kwds) {
    Py_ssize_t ctx_ptr, fb_ptr;
    int prepare = 1;
    static const char* kwlist[] = {"context", "framebuffer", "prepare", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn|p"), (char**)kwlist, &ctx_ptr, &fb_ptr, &prepare))
        return NULL;
    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    OffscreenFramebuffer* fb = reinterpret_cast<OffscreenFramebuffer*>(fb_ptr);
    if (!fb) {
        PyErr_SetString(PyExc_ValueError, "framebuffer is null");
        return NULL;
    }
    if (prepare && fb->color_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        const bool record_in_frame = ctx->in_frame;
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (record_in_frame) {
            end_active_render_pass(ctx);
            command = ctx->frames[ctx->current_frame].command_buffer;
        } else {
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = ctx->command_pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkResult result = vkAllocateCommandBuffers(ctx->device, &ai, &command);
            if (result != VK_SUCCESS) {
                PyErr_SetString(PyExc_RuntimeError, "Could not allocate camera texture transition");
                return NULL;
            }
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(command, &bi);
            if (result != VK_SUCCESS) {
                vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &command);
                PyErr_SetString(PyExc_RuntimeError, "Could not begin camera texture transition");
                return NULL;
            }
        }
        transition_image_layout(command, fb->color_image, fb->color_layout,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
        if (!record_in_frame) {
            VkResult result = vkEndCommandBuffer(command);
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            if (result == VK_SUCCESS)
                result = vkQueueSubmit(ctx->graphics_queue, 1, &submit, VK_NULL_HANDLE);
            if (result == VK_SUCCESS) {
                Py_BEGIN_ALLOW_THREADS
                result = vkQueueWaitIdle(ctx->graphics_queue);
                Py_END_ALLOW_THREADS
            }
            vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &command);
            if (result != VK_SUCCESS) {
                PyErr_SetString(PyExc_RuntimeError, "Could not submit camera texture transition");
                return NULL;
            }
        }
        fb->color_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        fb->color_texture.current_layout = fb->color_layout;
    }
    return PyLong_FromSsize_t(
        reinterpret_cast<Py_ssize_t>(&fb->color_texture));
}

extern "C"
PyObject* vulkan_destroy_offscreen_framebuffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr, fb_ptr;
    static const char* kwlist[] = {"context", "framebuffer", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist, &ctx_ptr, &fb_ptr))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    OffscreenFramebuffer* fb = reinterpret_cast<OffscreenFramebuffer*>(fb_ptr);
    if (!fb) Py_RETURN_NONE;
    if (ctx->live_offscreens.erase(fb) == 0)
        Py_RETURN_NONE;

    // A device-idle wait does not protect commands still being recorded.
    // Retire the attachments after this frame's temporary framebuffers,
    // when its fence guarantees that both recorded and submitted uses ended.
    if (ctx->active_offscreen == fb)
        ctx->active_offscreen = nullptr;
    ctx->pending_deletes[deferred_delete_slot(ctx)].offscreens.push_back(fb);
    Py_RETURN_NONE;
}

extern "C"
PyObject* vulkan_set_active_offscreen(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr, fb_ptr;
    static const char* kwlist[] = {"context", "framebuffer", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist, &ctx_ptr, &fb_ptr))
        return NULL;
    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    ctx->active_offscreen = reinterpret_cast<OffscreenFramebuffer*>(fb_ptr);
    return PyLong_FromLong(ctx->active_offscreen ? 1 : 0);
}

// Read back the color attachment of an offscreen framebuffer into a
// caller-provided buffer (a writable Python buffer, e.g. a numpy uint8
// array of shape (h, w, 4)). Performs a layout transition + image-to-buffer
// copy via a one-shot command buffer and waits for completion before
// returning. Output is row-major, top-down, RGBA8.
extern "C"
PyObject* vulkan_read_framebuffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr, fb_ptr;
    Py_buffer pybuf;
    static const char* kwlist[] = {"context", "framebuffer", "data", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nnw*"), (char**)kwlist,
            &ctx_ptr, &fb_ptr, &pybuf))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    OffscreenFramebuffer* fb = reinterpret_cast<OffscreenFramebuffer*>(fb_ptr);
    if (!fb) {
        PyBuffer_Release(&pybuf);
        PyErr_SetString(PyExc_ValueError, "framebuffer is null");
        return NULL;
    }
    const int pixel_bytes = fb->color_format == kColorFormat16 ? 8 : 4;
    Py_ssize_t needed = (Py_ssize_t)fb->width * (Py_ssize_t)fb->height * pixel_bytes;
    if (pybuf.len < needed) {
        PyBuffer_Release(&pybuf);
        PyErr_Format(PyExc_ValueError,
                     "Output buffer too small: have %zd need %zd",
                     pybuf.len, needed);
        return NULL;
    }

    // Staging buffer (host-visible).
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = (VkDeviceSize)needed;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo sinfo{};
    if (vmaCreateBuffer(ctx->allocator, &bi, &ai, &staging, &stagingAlloc, &sinfo) != VK_SUCCESS) {
        PyBuffer_Release(&pybuf);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create readback staging buffer");
        return NULL;
    }

    // One-shot command buffer for the image -> buffer copy.
    VkCommandBufferAllocateInfo cmd_ai{};
    cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool = ctx->command_pool;
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx->device, &cmd_ai, &cmd);

    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);

    // The render pass leaves the image in TRANSFER_SRC_OPTIMAL, so usually
    // no transition is needed -- but be defensive (capture may be called
    // without a frame having drawn to this FB yet).
    if (fb->color_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        transition_image_layout(cmd, fb->color_image,
                                fb->color_layout,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
        fb->color_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        fb->color_texture.current_layout = fb->color_layout;
    }

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {fb->width, fb->height, 1};
    vkCmdCopyImageToBuffer(cmd, fb->color_image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging, 1, &region);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx->graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->graphics_queue);

    // HOST_ACCESS_RANDOM steers VMA toward cached memory, which may be
    // non-coherent. Memory must be mapped before invalidation.
    void* mapped = sinfo.pMappedData;
    const bool mapped_here = mapped == nullptr;
    if (mapped_here &&
        vmaMapMemory(ctx->allocator, stagingAlloc, &mapped) != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
        vmaDestroyBuffer(ctx->allocator, staging, stagingAlloc);
        PyBuffer_Release(&pybuf);
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to map framebuffer readback buffer");
        return NULL;
    }
    vmaInvalidateAllocation(ctx->allocator, stagingAlloc, 0, VK_WHOLE_SIZE);
    memcpy(pybuf.buf, mapped, needed);
    if (mapped_here)
        vmaUnmapMemory(ctx->allocator, stagingAlloc);

    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
    vmaDestroyBuffer(ctx->allocator, staging, stagingAlloc);
    PyBuffer_Release(&pybuf);

    Py_RETURN_NONE;
}

// Read the snapshot copied immediately before the most recent presentation.
// OpenXR owns its swapchain images. Copy an owned, completed eye target into
// the acquired image, leaving it in the layout required by xrReleaseSwapchainImage.
extern "C"
PyObject* vulkan_copy_framebuffer_to_image(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr, fb_ptr;
    unsigned long long image_handle;
    int format;
    static const char* kwlist[] = {"context", "framebuffer", "image", "format", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "nnKi", (char**)kwlist,
                                     &ctx_ptr, &fb_ptr, &image_handle, &format))
        return NULL;
    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    OffscreenFramebuffer* fb = reinterpret_cast<OffscreenFramebuffer*>(fb_ptr);
    if (!fb || !ctx->live_offscreens.count(fb) || !image_handle || ctx->in_frame) {
        PyErr_SetString(PyExc_ValueError, "Complete the eye frame before copying to OpenXR");
        return NULL;
    }
    // A raw compatible-format copy preserves ChimeraX's gamma-encoded
    // colors. Vulkan rendering directly into an sRGB target would encode
    // them twice; the XR compositor performs the required sRGB decode.
    if (fb->color_format != VK_FORMAT_R8G8B8A8_UNORM ||
            format != VK_FORMAT_R8G8B8A8_SRGB) {
        PyErr_SetString(PyExc_ValueError, "OpenXR eye copy requires RGBA8 to sRGB RGBA8");
        return NULL;
    }
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = ctx->command_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer command;
    VkResult result = vkAllocateCommandBuffers(ctx->device, &ai, &command);
    if (result != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate OpenXR copy commands");
        return NULL;
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(command, &bi);
    if (result == VK_SUCCESS) {
        VkImage image = (VkImage)(uintptr_t)image_handle;
        transition_image_layout(command, fb->color_image, fb->color_layout,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition_image_layout(command, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageCopy region{};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.layerCount = 1;
        region.dstSubresource = region.srcSubresource;
        region.extent = {fb->width, fb->height, 1};
        vkCmdCopyImage(command, fb->color_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        transition_image_layout(command, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        result = vkEndCommandBuffer(command);
    }
    if (result == VK_SUCCESS) {
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(ctx->graphics_queue, 1, &submit, VK_NULL_HANDLE);
        if (result == VK_SUCCESS) {
            Py_BEGIN_ALLOW_THREADS
            result = vkQueueWaitIdle(ctx->graphics_queue);
            Py_END_ALLOW_THREADS
        }
    }
    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &command);
    if (result != VK_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "OpenXR eye copy failed (VkResult %d)", (int)result);
        return NULL;
    }
    fb->color_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    fb->color_texture.current_layout = fb->color_layout;
    Py_RETURN_NONE;
}

// Read the snapshot copied immediately before the most recent presentation.
// A presented swapchain image is owned by the presentation engine and cannot
// legally be transitioned for a direct copy until it is acquired again.
extern "C"
PyObject* vulkan_read_screen(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    Py_buffer pybuf;
    static const char* kwlist[] = {"context", "data", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nw*"), (char**)kwlist, &ctx_ptr, &pybuf))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    const uint32_t index = ctx->last_presented_image_index;
    const uint32_t width = ctx->swapchain_extent.width;
    const uint32_t height = ctx->swapchain_extent.height;
    const Py_ssize_t needed =
        (Py_ssize_t)width * (Py_ssize_t)height * 4;
    if (index >= ctx->screen_readback_buffers.size() ||
        index >= ctx->screen_readback_allocations.size() ||
        index >= ctx->swapchain_image_used.size() ||
        !ctx->swapchain_image_used[index] || pybuf.len < needed) {
        PyBuffer_Release(&pybuf);
        PyErr_SetString(PyExc_ValueError, "No readable Vulkan screen image");
        return NULL;
    }

    // Finish the copy recorded by end_frame.  The current frame, if one is
    // being recorded, has not been submitted and therefore cannot deadlock
    // this wait.
    vkQueueWaitIdle(ctx->graphics_queue);
    VmaAllocation allocation = ctx->screen_readback_allocations[index];
    void* mapped = nullptr;
    if (vmaMapMemory(ctx->allocator, allocation, &mapped) != VK_SUCCESS) {
        PyBuffer_Release(&pybuf);
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to map Vulkan screen snapshot");
        return NULL;
    }
    vmaInvalidateAllocation(ctx->allocator, allocation, 0, VK_WHOLE_SIZE);
    // Preserve the renderer API's RGBA contract.  The preferred and fallback
    // swapchain formats are BGRA8, but handle an implementation-selected
    // RGBA8 format as well.
    const unsigned char* source = static_cast<const unsigned char*>(mapped);
    unsigned char* destination = static_cast<unsigned char*>(pybuf.buf);
    const bool bgra =
        ctx->swapchain_format == VK_FORMAT_B8G8R8A8_UNORM ||
        ctx->swapchain_format == VK_FORMAT_B8G8R8A8_SRGB;
    if (bgra) {
        for (Py_ssize_t p = 0; p < needed; p += 4) {
            destination[p] = source[p + 2];
            destination[p + 1] = source[p + 1];
            destination[p + 2] = source[p];
            destination[p + 3] = source[p + 3];
        }
    } else {
        memcpy(destination, source, needed);
    }
    vmaUnmapMemory(ctx->allocator, allocation);
    PyBuffer_Release(&pybuf);
    Py_RETURN_NONE;
}
