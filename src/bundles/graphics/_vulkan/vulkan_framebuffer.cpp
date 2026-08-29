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

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// Build the render pass that the offscreen framebuffer's draws use.
// The color attachment ends in TRANSFER_SRC_OPTIMAL so the readback copy
// can happen without an extra layout transition.
VkRenderPass create_offscreen_render_pass(VkDevice device) {
    VkAttachmentDescription attachments[2] = {};

    attachments[0].format = kColorFormat;
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
    static const char* kwlist[] = {"context", "width", "height", "alpha", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nii|i"), (char**)kwlist,
            &ctx_ptr, &width, &height, &alpha))
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
    fb->alpha = (alpha != 0);
    fb->color_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (!create_image(ctx, fb->width, fb->height, kColorFormat,
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
                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                      VK_IMAGE_ASPECT_DEPTH_BIT,
                      &fb->depth_image, &fb->depth_alloc, &fb->depth_view)) {
        destroy_offscreen(ctx, fb);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create offscreen depth image");
        return NULL;
    }

    fb->render_pass = create_offscreen_render_pass(ctx->device);
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

    ctx->live_offscreens.insert(fb);
    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(fb));
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

    // Make sure the GPU isn't still referencing this framebuffer before we
    // tear it down. Image capture is a one-shot off the hot path, so the
    // device-wide wait here is fine. Drain deferred deletes too: per-pass
    // framebuffers recorded during the capture (offscreen load/clear
    // variants, OIT) reference this target's color/depth views.
    vkDeviceWaitIdle(ctx->device);
    flush_all_pending_deletes(ctx);
    if (ctx->active_offscreen == fb)
        ctx->active_offscreen = nullptr;
    destroy_offscreen(ctx, fb);
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

    Py_ssize_t needed = (Py_ssize_t)fb->width * (Py_ssize_t)fb->height * 4;
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
    // non-coherent; invalidate so the CPU reads the GPU's writes rather
    // than stale cache lines. No-op on coherent memory.
    vmaInvalidateAllocation(ctx->allocator, stagingAlloc, 0, VK_WHOLE_SIZE);

    // Map (already mapped via VMA flag) and copy out.
    if (sinfo.pMappedData) {
        memcpy(pybuf.buf, sinfo.pMappedData, needed);
    } else {
        void* mapped = nullptr;
        vmaMapMemory(ctx->allocator, stagingAlloc, &mapped);
        memcpy(pybuf.buf, mapped, needed);
        vmaUnmapMemory(ctx->allocator, stagingAlloc);
    }

    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &cmd);
    vmaDestroyBuffer(ctx->allocator, staging, stagingAlloc);
    PyBuffer_Release(&pybuf);

    Py_RETURN_NONE;
}
