/*
 * Vulkan context management: instance, device, swapchain, render passes,
 * frame lifecycle, and descriptor infrastructure.
 *
 * Uses vk-bootstrap for instance/device/swapchain creation and VMA for
 * GPU memory allocation.  VMA_IMPLEMENTATION is defined here so that
 * the VMA allocator is compiled exactly once.
 */

#include <Python.h>

#define VMA_IMPLEMENTATION
#include "vulkan_internal.h"

#include <VkBootstrap.h>

#include <cstring>
#include <algorithm>
#include <utility>

#if defined(_WIN32)
  #include <windows.h>
  #include <vulkan/vulkan_win32.h>
#elif defined(__APPLE__)
  // Safe to include from C++ — CAMetalLayer is typedef'd to void when
  // __OBJC__ is not defined. We only need the extension name macros.
  #include <vulkan/vulkan_metal.h>
#endif

// ---------------------------------------------------------------------------
//  Internal helpers (forward-declared in vulkan_internal.h)
// ---------------------------------------------------------------------------

void transition_image_layout(VkCommandBuffer cmd, VkImage image,
                             VkImageLayout old_layout, VkImageLayout new_layout,
                             VkImageAspectFlags aspect)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    // Determine access masks from layouts.
    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = 0;
    } else if (old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    if (new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (new_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
        barrier.dstAccessMask = 0;
        dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
}

// ---------------------------------------------------------------------------

void create_depth_resources(VulkanContext* ctx)
{
    VkImageCreateInfo img_info{};
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.format = VK_FORMAT_D32_SFLOAT;
    img_info.extent = { ctx->swapchain_extent.width, ctx->swapchain_extent.height, 1 };
    img_info.mipLevels = 1;
    img_info.arrayLayers = 1;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_SRC: the silhouette pass copies the depth into a separate
    // sampleable texture (the live depth can't be sampled while attached).
    img_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(ctx->allocator, &img_info, &alloc_info,
                   &ctx->depth_image.image, &ctx->depth_image.allocation, nullptr);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = ctx->depth_image.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_D32_SFLOAT;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    vkCreateImageView(ctx->device, &view_info, nullptr, &ctx->depth_image.image_view);

    ctx->depth_image.format = VK_FORMAT_D32_SFLOAT;
    ctx->depth_image.width = ctx->swapchain_extent.width;
    ctx->depth_image.height = ctx->swapchain_extent.height;
    ctx->depth_image.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ctx->depth_image.sampler = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------

void create_main_render_pass(VulkanContext* ctx)
{
    // ---- Main render pass: color + depth ----
    {
        VkAttachmentDescription attachments[2] = {};

        // Color attachment
        attachments[0].format = ctx->swapchain_format;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // Depth attachment
        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
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

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = 2;
        rp_info.pAttachments = attachments;
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 1;
        rp_info.pDependencies = &dependency;

        vkCreateRenderPass(ctx->device, &rp_info, nullptr, &ctx->main_render_pass);
    }

    // ---- Depth-only render pass ----
    {
        VkAttachmentDescription depth_att{};
        depth_att.format = VK_FORMAT_D32_SFLOAT;
        depth_att.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        // Transition to shader-read at the end of the pass so subsequent
        // draws can sample the depth texture as a shadow map without an
        // explicit barrier.
        depth_att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 0;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 0;
        subpass.pColorAttachments = nullptr;
        subpass.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency dependencies[2] = {};
        // Wait for any previous depth use to finish before this pass writes.
        // FRAGMENT_SHADER is included so re-rendering a shadow map orders the
        // prior lighting pass's *reads* of it before the clear (WAR hazard);
        // the fragment-test stages alone don't cover sampled reads.
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        // Make depth writes visible to subsequent fragment-shader reads
        // (i.e. when the rendered shadow map is sampled in the lighting pass).
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = 1;
        rp_info.pAttachments = &depth_att;
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 2;
        rp_info.pDependencies = dependencies;

        vkCreateRenderPass(ctx->device, &rp_info, nullptr, &ctx->depth_only_render_pass);
    }
}

// ---------------------------------------------------------------------------

bool create_swapchain(VulkanContext* ctx, uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder builder{ctx->physical_device, ctx->device, ctx->surface};

#if defined(__APPLE__)
    // PASS_THROUGH (VK_EXT_swapchain_colorspace) leaves the CAMetalLayer
    // colorspace unset so the compositor treats our pixels like the
    // OpenGL view's untagged drawable — identical colors when switching
    // renderers. Only advertised when the instance extension was enabled;
    // the sRGB fallback keeps swapchain creation working regardless.
    builder.set_desired_format({VK_FORMAT_B8G8R8A8_UNORM,
                                VK_COLOR_SPACE_PASS_THROUGH_EXT})
           .add_fallback_format({VK_FORMAT_B8G8R8A8_UNORM,
                                 VK_COLOR_SPACE_SRGB_NONLINEAR_KHR});
#else
    builder.set_desired_format({VK_FORMAT_B8G8R8A8_UNORM,
                                VK_COLOR_SPACE_SRGB_NONLINEAR_KHR});
#endif

    auto swap_ret = builder
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .set_desired_min_image_count(2)
        .build();

    if (!swap_ret) {
        return false;
    }

    vkb::Swapchain vkb_swap = swap_ret.value();
    ctx->swapchain = vkb_swap.swapchain;
    ctx->swapchain_format = vkb_swap.image_format;
    ctx->swapchain_extent = vkb_swap.extent;
    ctx->swapchain_images = vkb_swap.get_images().value();
    ctx->swapchain_image_views = vkb_swap.get_image_views().value();

    // One render-finished semaphore per swapchain image -- see comment
    // in vulkan_internal.h. Created here (after image count is known)
    // and torn down in cleanup_swapchain.
    {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        ctx->render_finished_per_image.resize(ctx->swapchain_images.size());
        for (size_t i = 0; i < ctx->render_finished_per_image.size(); ++i) {
            vkCreateSemaphore(ctx->device, &si, nullptr,
                              &ctx->render_finished_per_image[i]);
        }
    }

    // Fresh images have never been rendered to; a load_color pass on one
    // must clear instead of loading undefined contents.
    ctx->swapchain_image_used.assign(ctx->swapchain_images.size(), 0);

    ctx->width = ctx->swapchain_extent.width;
    ctx->height = ctx->swapchain_extent.height;

    // Create depth resources sized to match the swapchain.
    create_depth_resources(ctx);

    // Create (or recreate) the main render pass and framebuffers.
    create_main_render_pass(ctx);
    create_swapchain_framebuffers(ctx);

    return true;
}

// ---------------------------------------------------------------------------

void cleanup_swapchain(VulkanContext* ctx)
{
    for (auto fb : ctx->swapchain_framebuffers) {
        vkDestroyFramebuffer(ctx->device, fb, nullptr);
    }
    ctx->swapchain_framebuffers.clear();

    if (ctx->main_render_pass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(ctx->device, ctx->main_render_pass, nullptr);
        ctx->main_render_pass = VK_NULL_HANDLE;
    }
    if (ctx->depth_only_render_pass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(ctx->device, ctx->depth_only_render_pass, nullptr);
        ctx->depth_only_render_pass = VK_NULL_HANDLE;
    }

    // Depth image
    if (ctx->depth_image.image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx->device, ctx->depth_image.image_view, nullptr);
        ctx->depth_image.image_view = VK_NULL_HANDLE;
    }
    if (ctx->depth_image.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(ctx->device, ctx->depth_image.sampler, nullptr);
        ctx->depth_image.sampler = VK_NULL_HANDLE;
    }
    if (ctx->depth_image.image != VK_NULL_HANDLE) {
        vmaDestroyImage(ctx->allocator, ctx->depth_image.image, ctx->depth_image.allocation);
        ctx->depth_image.image = VK_NULL_HANDLE;
        ctx->depth_image.allocation = VK_NULL_HANDLE;
    }

    for (auto iv : ctx->swapchain_image_views) {
        vkDestroyImageView(ctx->device, iv, nullptr);
    }
    ctx->swapchain_image_views.clear();
    ctx->swapchain_images.clear();

    for (auto sem : ctx->render_finished_per_image) {
        if (sem != VK_NULL_HANDLE)
            vkDestroySemaphore(ctx->device, sem, nullptr);
    }
    ctx->render_finished_per_image.clear();

    if (ctx->swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(ctx->device, ctx->swapchain, nullptr);
        ctx->swapchain = VK_NULL_HANDLE;
    }
}

// Move the active presentation objects out of the context without
// destroying them.  This is only called between frames, after Python has
// ended any render pass/command buffer targeting the current window.
static void take_window_swapchain(VulkanContext* ctx,
                                  WindowSwapchainState& state)
{
    state.surface = ctx->surface;
    state.owns_surface = ctx->owns_surface;
    state.swapchain = ctx->swapchain;
    state.swapchain_format = ctx->swapchain_format;
    state.swapchain_extent = ctx->swapchain_extent;
    state.swapchain_images.swap(ctx->swapchain_images);
    state.swapchain_image_views.swap(ctx->swapchain_image_views);
    state.render_finished_per_image.swap(ctx->render_finished_per_image);
    state.swapchain_image_used.swap(ctx->swapchain_image_used);
    state.depth_image = ctx->depth_image;
    state.main_render_pass = ctx->main_render_pass;
    state.depth_only_render_pass = ctx->depth_only_render_pass;
    state.swapchain_framebuffers.swap(ctx->swapchain_framebuffers);
    state.width = ctx->width;
    state.height = ctx->height;

    ctx->surface = VK_NULL_HANDLE;
    ctx->owns_surface = false;
    ctx->swapchain = VK_NULL_HANDLE;
    ctx->swapchain_format = VK_FORMAT_UNDEFINED;
    ctx->swapchain_extent = {};
    ctx->depth_image = {};
    ctx->main_render_pass = VK_NULL_HANDLE;
    ctx->depth_only_render_pass = VK_NULL_HANDLE;
    ctx->width = 0;
    ctx->height = 0;
}

static void restore_window_swapchain(VulkanContext* ctx,
                                     WindowSwapchainState& state)
{
    ctx->surface = state.surface;
    ctx->owns_surface = state.owns_surface;
    ctx->swapchain = state.swapchain;
    ctx->swapchain_format = state.swapchain_format;
    ctx->swapchain_extent = state.swapchain_extent;
    ctx->swapchain_images.swap(state.swapchain_images);
    ctx->swapchain_image_views.swap(state.swapchain_image_views);
    ctx->render_finished_per_image.swap(state.render_finished_per_image);
    ctx->swapchain_image_used.swap(state.swapchain_image_used);
    ctx->depth_image = state.depth_image;
    ctx->main_render_pass = state.main_render_pass;
    ctx->depth_only_render_pass = state.depth_only_render_pass;
    ctx->swapchain_framebuffers.swap(state.swapchain_framebuffers);
    ctx->width = state.width;
    ctx->height = state.height;

    state.surface = VK_NULL_HANDLE;
    state.owns_surface = false;
    state.swapchain = VK_NULL_HANDLE;
    state.depth_image = {};
    state.main_render_pass = VK_NULL_HANDLE;
    state.depth_only_render_pass = VK_NULL_HANDLE;
}

static void destroy_window_swapchain(VulkanContext* ctx,
                                     WindowSwapchainState& state)
{
    for (VkFramebuffer fb : state.swapchain_framebuffers)
        vkDestroyFramebuffer(ctx->device, fb, nullptr);
    if (state.main_render_pass != VK_NULL_HANDLE)
        vkDestroyRenderPass(ctx->device, state.main_render_pass, nullptr);
    if (state.depth_only_render_pass != VK_NULL_HANDLE)
        vkDestroyRenderPass(ctx->device, state.depth_only_render_pass, nullptr);

    if (state.depth_image.image_view != VK_NULL_HANDLE)
        vkDestroyImageView(ctx->device, state.depth_image.image_view, nullptr);
    if (state.depth_image.sampler != VK_NULL_HANDLE)
        vkDestroySampler(ctx->device, state.depth_image.sampler, nullptr);
    if (state.depth_image.image != VK_NULL_HANDLE)
        vmaDestroyImage(ctx->allocator, state.depth_image.image,
                        state.depth_image.allocation);

    for (VkImageView view : state.swapchain_image_views)
        vkDestroyImageView(ctx->device, view, nullptr);
    for (VkSemaphore semaphore : state.render_finished_per_image) {
        if (semaphore != VK_NULL_HANDLE)
            vkDestroySemaphore(ctx->device, semaphore, nullptr);
    }
    if (state.swapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(ctx->device, state.swapchain, nullptr);
    if (state.owns_surface && state.surface != VK_NULL_HANDLE)
        vkDestroySurfaceKHR(ctx->instance, state.surface, nullptr);
}

static bool recreate_active_swapchain(VulkanContext* ctx,
                                      uint32_t width, uint32_t height)
{
    vkDeviceWaitIdle(ctx->device);
    flush_all_pending_deletes(ctx);
    destroy_oit_resources_internal(ctx);
    cleanup_swapchain(ctx);
    return create_swapchain(ctx, width, height);
}

// ---------------------------------------------------------------------------

void create_swapchain_framebuffers(VulkanContext* ctx)
{
    ctx->swapchain_framebuffers.resize(ctx->swapchain_image_views.size());

    for (size_t i = 0; i < ctx->swapchain_image_views.size(); ++i) {
        VkImageView attachments[] = {
            ctx->swapchain_image_views[i],
            ctx->depth_image.image_view
        };

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = ctx->main_render_pass;
        fb_info.attachmentCount = 2;
        fb_info.pAttachments = attachments;
        fb_info.width = ctx->swapchain_extent.width;
        fb_info.height = ctx->swapchain_extent.height;
        fb_info.layers = 1;

        vkCreateFramebuffer(ctx->device, &fb_info, nullptr,
                            &ctx->swapchain_framebuffers[i]);
    }
}

// ---------------------------------------------------------------------------

void create_descriptor_infrastructure(VulkanContext* ctx)
{
    // ---- Set 0: Mixed UBO/SSBO bindings (see set0_descriptor_type) ----
    {
        VK_TRACE("create_descriptor_infrastructure: %u set-0 bindings", (unsigned)MAX_UBO_BINDINGS);
        VkDescriptorSetLayoutBinding bindings[MAX_UBO_BINDINGS];
        for (uint32_t i = 0; i < MAX_UBO_BINDINGS; ++i) {
            bindings[i] = {};
            bindings[i].binding = i;
            bindings[i].descriptorType = set0_descriptor_type(i);
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                     VK_SHADER_STAGE_FRAGMENT_BIT;
            VK_TRACE("  binding %u type=%d (%s)", i, (int)bindings[i].descriptorType,
                     bindings[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? "SSBO" : "UBO");
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = MAX_UBO_BINDINGS;
        layout_info.pBindings = bindings;

        vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr,
                                    &ctx->descriptor_set_layout_ubo);
    }

    // ---- Set 1: Combined image samplers (bindings 0..7) ----
    {
        VkDescriptorSetLayoutBinding bindings[MAX_TEXTURE_BINDINGS];
        for (uint32_t i = 0; i < MAX_TEXTURE_BINDINGS; ++i) {
            bindings[i] = {};
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = MAX_TEXTURE_BINDINGS;
        layout_info.pBindings = bindings;

        vkCreateDescriptorSetLayout(ctx->device, &layout_info, nullptr,
                                    &ctx->descriptor_set_layout_textures);
    }

    // ---- Pipeline layout (set 0 = UBO, set 1 = textures) ----
    {
        VkDescriptorSetLayout set_layouts[] = {
            ctx->descriptor_set_layout_ubo,
            ctx->descriptor_set_layout_textures
        };

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = set_layouts;

        vkCreatePipelineLayout(ctx->device, &layout_info, nullptr,
                               &ctx->shared_pipeline_layout);
    }

    // ---- Per-frame descriptor pools ----
    // Each frame in flight gets its own pool, sized to hold many descriptor
    // sets. flush_descriptors allocates fresh sets every call so previously
    // recorded bindings stay valid; the pool is reset at the start of each
    // frame. Sizes are generous to handle deeply-tiled passes (multishadow).
    {
        // Count UBO vs SSBO bindings based on the layout helper.
        uint32_t ubo_count = 0;
        uint32_t ssbo_count = 0;
        for (uint32_t i = 0; i < MAX_UBO_BINDINGS; ++i) {
            if (set0_descriptor_type(i) == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                ssbo_count++;
            else
                ubo_count++;
        }

        constexpr uint32_t MAX_SETS_PER_FRAME = 2048;  // 2 sets × ~1024 flushes

        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              ubo_count * (MAX_SETS_PER_FRAME / 2) },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              ssbo_count * (MAX_SETS_PER_FRAME / 2) },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              MAX_TEXTURE_BINDINGS * (MAX_SETS_PER_FRAME / 2) },
        };

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = 0;  // pool reset rather than per-set free
        pool_info.maxSets = MAX_SETS_PER_FRAME;
        pool_info.poolSizeCount = 3;
        pool_info.pPoolSizes = pool_sizes;

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkCreateDescriptorPool(ctx->device, &pool_info, nullptr,
                                   &ctx->descriptor_pools[i]);
        }
    }

    // ---- Pipeline cache ----
    {
        VkPipelineCacheCreateInfo cache_info{};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        vkCreatePipelineCache(ctx->device, &cache_info, nullptr, &ctx->pipeline_cache);
    }
}

// ---------------------------------------------------------------------------

void end_active_render_pass(VulkanContext* ctx)
{
    if (ctx->in_render_pass) {
        vkCmdEndRenderPass(ctx->frames[ctx->current_frame].command_buffer);
        ctx->in_render_pass = false;
    }
    // Once the active pass is ended, every entry still on the stack is dead;
    // queue owned objects so they're destroyed after the GPU is done with
    // this frame's commands. Dropping them here (as the implicit-end paths
    // used to) leaks one VkRenderPass + VkFramebuffer per pass.
    if (!ctx->render_pass_stack.empty()) {
        PendingDeletes& pd = ctx->pending_deletes[deferred_delete_slot(ctx)];
        for (RenderPassState& state : ctx->render_pass_stack) {
            if (state.owns_framebuffer && state.framebuffer != VK_NULL_HANDLE)
                pd.framebuffers.push_back(state.framebuffer);
            if (state.owns_render_pass && state.render_pass != VK_NULL_HANDLE)
                pd.render_passes.push_back(state.render_pass);
        }
        ctx->render_pass_stack.clear();
    }
}

// ===========================================================================
//  Public API — extern "C" functions matching vulkan_api.h
// ===========================================================================

// Helper: build a VkInstance with the right extensions for the platform.
// Returns a vkb::Instance or sets a Python error.
static bool build_vulkan_instance(vkb::Instance& out_inst) {
    // Pass vkGetInstanceProcAddr directly so vk-bootstrap doesn't try to
    // dlopen("libvulkan.dylib") by bare name — that fails on macOS when
    // the library isn't on a standard search path.
    vkb::InstanceBuilder inst_builder{vkGetInstanceProcAddr};
    inst_builder.set_app_name("ChimeraX")
                .require_api_version(1, 2, 0)
                .enable_extension(VK_KHR_SURFACE_EXTENSION_NAME);

#if defined(_WIN32)
    inst_builder.enable_extension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(__APPLE__)
    inst_builder.enable_extension(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
    // VK_KHR_portability_enumeration and its create flag are handled
    // automatically by vk-bootstrap when the extension is available.

    // VK_EXT_swapchain_colorspace allows requesting PASS_THROUGH for the
    // swapchain so MoltenVK leaves the CAMetalLayer's colorspace unset.
    // With the default SRGB_NONLINEAR colorspace the macOS compositor
    // color-matches the sRGB-tagged layer into the display's native
    // (typically P3) gamut, which visibly desaturates the image relative
    // to the OpenGL view, whose untagged drawable is composited raw.
    {
        auto sysinfo = vkb::SystemInfo::get_system_info(vkGetInstanceProcAddr);
        if (sysinfo &&
            sysinfo->is_extension_available("VK_EXT_swapchain_colorspace"))
            inst_builder.enable_extension("VK_EXT_swapchain_colorspace");
    }
#endif

#if !defined(__APPLE__) && !defined(NDEBUG)
    // MoltenVK does not support the standard Vulkan validation layers.
    // Debug builds only: validation costs real performance and the
    // messenger handle is only tracked/destroyed when NDEBUG is unset.
    inst_builder.request_validation_layers()
                .use_default_debug_messenger();
#endif

    auto inst_ret = inst_builder.build();
    if (!inst_ret) {
        PyErr_Format(PyExc_RuntimeError, "Failed to create Vulkan instance: %s",
                     inst_ret.error().message().c_str());
        return false;
    }
    out_inst = inst_ret.value();
    return true;
}

// ---------------------------------------------------------------------------
//  Create a VkInstance and return its handle. Used on macOS so Python can
//  pass it to _metal_rt.create_surface before calling create_context.
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_create_instance(PyObject* self, PyObject* args, PyObject* kwds)
{
    vkb::Instance vkb_inst{};
    if (!build_vulkan_instance(vkb_inst))
        return NULL;
    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(vkb_inst.instance));
}

extern "C"
PyObject* vulkan_create_context(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t window_handle;
    Py_ssize_t instance_handle = 0;  // Optional pre-created VkInstance (macOS)
    Py_ssize_t surface_handle = 0;   // Optional pre-created VkSurfaceKHR (macOS)
    int owns_instance_arg = -1;
    int owns_surface_arg = -1;
    static const char* kwlist[] = {"window_handle", "instance", "surface",
                                   "owns_instance", "owns_surface", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n|nnii"), (char**)kwlist,
            &window_handle, &instance_handle, &surface_handle,
            &owns_instance_arg, &owns_surface_arg))
        return NULL;

    const bool owns_instance = owns_instance_arg >= 0
        ? owns_instance_arg != 0 : instance_handle == 0;
    const bool owns_surface = owns_surface_arg >= 0
        ? owns_surface_arg != 0 : surface_handle == 0;

    // ---- Instance ----
    vkb::Instance vkb_inst{};
    if (instance_handle) {
        // Wrap a pre-created instance for vk-bootstrap's device selection.
        // The instance was created by vulkan_create_instance above.
        vkb_inst.instance = reinterpret_cast<VkInstance>(instance_handle);
        // vkb_inst.fp_vkGetInstanceProcAddr is needed by the device selector.
        // vk-bootstrap populates it from vkGetInstanceProcAddr.
        vkb_inst.fp_vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    } else {
        if (!build_vulkan_instance(vkb_inst))
            return NULL;
    }

    // ---- Surface ----
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    if (surface_handle) {
        surface = reinterpret_cast<VkSurfaceKHR>(surface_handle);
    }
#if defined(_WIN32)
    else {
        VkWin32SurfaceCreateInfoKHR surf_info = {};
        surf_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        surf_info.hinstance = GetModuleHandle(nullptr);
        surf_info.hwnd = reinterpret_cast<HWND>(window_handle);
        if (vkCreateWin32SurfaceKHR(vkb_inst.instance, &surf_info, nullptr, &surface) != VK_SUCCESS) {
            if (owns_instance)
                vkb::destroy_instance(vkb_inst);
            PyErr_SetString(PyExc_RuntimeError, "vkCreateWin32SurfaceKHR failed");
            return NULL;
        }
    }
#else
    else {
        if (owns_instance)
            vkb::destroy_instance(vkb_inst);
        PyErr_SetString(PyExc_RuntimeError,
                        "No pre-created surface and no platform surface creator");
        return NULL;
    }
#endif

    auto destroy_owned_surface_and_instance = [&]() {
        if (owns_surface && surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(vkb_inst.instance, surface, nullptr);
        if (owns_instance)
            vkb::destroy_instance(vkb_inst);
    };

    // ---- Physical device ----
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    auto phys_ret = selector
        .set_surface(surface)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
        .require_dedicated_transfer_queue()
        .select();

    if (!phys_ret) {
        // Fall back without requiring a dedicated transfer queue.
        selector = vkb::PhysicalDeviceSelector{vkb_inst};
        phys_ret = selector
            .set_surface(surface)
            .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
            .select();

        if (!phys_ret) {
            destroy_owned_surface_and_instance();
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to select Vulkan physical device: %s",
                         phys_ret.error().message().c_str());
            return NULL;
        }
    }

    vkb::PhysicalDevice vkb_phys = phys_ret.value();

#ifdef __APPLE__
    // MoltenVK devices advertise the portability subset extension; accept it
    // so vk-bootstrap doesn't reject the device.
    vkb_phys.enable_extensions_if_present({
        "VK_KHR_portability_subset",
    });
#endif

    // ---- Optional ray-tracing extensions ----
    // Probe and add VK_KHR_acceleration_structure + VK_KHR_ray_query as
    // *desired* extensions. If the device doesn't expose them we fall
    // through to a non-RT logical device instead of failing.
    bool rt_available = vkb_phys.enable_extensions_if_present({
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    });

#ifdef __APPLE__
    // VK_EXT_metal_objects lets us export the underlying MTLDevice and
    // MTLCommandQueue for native Metal ray tracing via the _metal_rt module.
    bool metal_objects_available = vkb_phys.enable_extensions_if_present({
        VK_EXT_METAL_OBJECTS_EXTENSION_NAME,
    });
#endif

    // ---- Logical device ----
    vkb::DeviceBuilder dev_builder{vkb_phys};

    // Feature chain for ray tracing. These structs must outlive the
    // device build call, so they're declared at function scope.
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
    as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    as_features.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR rq_features{};
    rq_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rq_features.rayQuery = VK_TRUE;

    // Required by VK_KHR_acceleration_structure: bufferDeviceAddress so the
    // BVH can reference its scratch/data buffers via GPU addresses.
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    if (rt_available) {
        features12.bufferDeviceAddress = VK_TRUE;
        // Descriptor indexing is also required by AS for descriptor binding
        // of variable-length data; enabling the umbrella feature is enough.
        features12.descriptorIndexing = VK_TRUE;
        dev_builder.add_pNext(&features12)
                   .add_pNext(&as_features)
                   .add_pNext(&rq_features);
    }

    auto dev_ret = dev_builder.build();
    if (!dev_ret) {
        destroy_owned_surface_and_instance();
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to create Vulkan logical device: %s",
                     dev_ret.error().message().c_str());
        return NULL;
    }
    vkb::Device vkb_dev = dev_ret.value();

    // ---- Allocate and populate context ----
    // Value-initialization zeroes POD members and default-constructs the
    // std::vector/std::string members. Do NOT memset over this — that would
    // corrupt the constructed containers and trip heap corruption later.
    VulkanContext* ctx = new VulkanContext();

    ctx->instance = vkb_inst.instance;
    ctx->surface = surface;
    ctx->owns_instance = owns_instance;
    ctx->owns_surface = owns_surface;
    ctx->current_window_handle = static_cast<std::uintptr_t>(window_handle);
    ctx->physical_device = vkb_phys.physical_device;
    ctx->device = vkb_dev.device;

    auto gq = vkb_dev.get_queue(vkb::QueueType::graphics);
    auto pq = vkb_dev.get_queue(vkb::QueueType::present);
    if (!gq || !pq) {
        delete ctx;
        vkb::destroy_device(vkb_dev);
        destroy_owned_surface_and_instance();
        PyErr_SetString(PyExc_RuntimeError, "Failed to get Vulkan queues");
        return NULL;
    }
    ctx->graphics_queue = gq.value();
    ctx->present_queue = pq.value();
    ctx->graphics_family = vkb_dev.get_queue_index(vkb::QueueType::graphics).value();
    ctx->present_family = vkb_dev.get_queue_index(vkb::QueueType::present).value();

    // Store the device name and a few physical-device limits we expose
    // to Python (max 3D texture size for volume rendering, etc.).
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &props);
    ctx->device_name = props.deviceName;
    ctx->max_3d_texture_size = props.limits.maxImageDimension3D;

    // ---- Ray-tracing function pointers ----
    // KHR extensions aren't statically exported on most loaders; resolve
    // through vkGetDeviceProcAddr so we don't need to link against an SDK
    // build that exports them. Mark the context as RT-capable only when
    // *all* required entry points resolved successfully.
    ctx->raytracing_supported = false;
    ctx->pfn_get_as_build_sizes = nullptr;
    ctx->pfn_create_as = nullptr;
    ctx->pfn_destroy_as = nullptr;
    ctx->pfn_cmd_build_as = nullptr;
    ctx->pfn_get_as_device_address = nullptr;
    if (rt_available) {
        ctx->pfn_get_as_build_sizes =
            (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(
                ctx->device, "vkGetAccelerationStructureBuildSizesKHR");
        ctx->pfn_create_as =
            (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(
                ctx->device, "vkCreateAccelerationStructureKHR");
        ctx->pfn_destroy_as =
            (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(
                ctx->device, "vkDestroyAccelerationStructureKHR");
        ctx->pfn_cmd_build_as =
            (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(
                ctx->device, "vkCmdBuildAccelerationStructuresKHR");
        ctx->pfn_get_as_device_address =
            (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(
                ctx->device, "vkGetAccelerationStructureDeviceAddressKHR");
        ctx->raytracing_supported =
            ctx->pfn_get_as_build_sizes && ctx->pfn_create_as &&
            ctx->pfn_destroy_as && ctx->pfn_cmd_build_as &&
            ctx->pfn_get_as_device_address;
    }
    ctx->min_as_scratch_alignment = 0;
    if (ctx->raytracing_supported) {
        // AS build scratch buffers must be aligned to this; commonly 128
        // or 256 bytes, larger than VMA's default buffer alignment.
        VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props{};
        as_props.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &as_props;
        vkGetPhysicalDeviceProperties2(ctx->physical_device, &props2);
        ctx->min_as_scratch_alignment =
            as_props.minAccelerationStructureScratchOffsetAlignment;
    }
    VK_TRACE("ray tracing %s", ctx->raytracing_supported ? "ENABLED" : "disabled");

#ifdef __APPLE__
    ctx->metal_objects_supported = metal_objects_available;
    VK_TRACE("VK_EXT_metal_objects %s",
             ctx->metal_objects_supported ? "ENABLED" : "disabled");
#else
    ctx->metal_objects_supported = false;
#endif

#ifndef NDEBUG
    ctx->debug_messenger = owns_instance
        ? vkb_inst.debug_messenger : VK_NULL_HANDLE;
#endif

    // ---- VMA allocator ----
    {
        VmaAllocatorCreateInfo alloc_info{};
        alloc_info.physicalDevice = ctx->physical_device;
        alloc_info.device = ctx->device;
        alloc_info.instance = ctx->instance;
        // Tell VMA about bufferDeviceAddress so it allocates AS / scratch
        // buffers with the BUFFER_DEVICE_ADDRESS_BIT memory flag and we
        // can call vkGetBufferDeviceAddress on them.
        if (ctx->raytracing_supported) {
            alloc_info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        }

        VkResult vr = vmaCreateAllocator(&alloc_info, &ctx->allocator);
        if (vr != VK_SUCCESS) {
            delete ctx;
            vkb::destroy_device(vkb_dev);
            destroy_owned_surface_and_instance();
            PyErr_SetString(PyExc_RuntimeError, "Failed to create VMA allocator");
            return NULL;
        }
    }

    // ---- Swapchain ----
    // Use a small default size; the caller will resize immediately.
    if (!create_swapchain(ctx, 1, 1)) {
        vmaDestroyAllocator(ctx->allocator);
        delete ctx;
        vkb::destroy_device(vkb_dev);
        destroy_owned_surface_and_instance();
        PyErr_SetString(PyExc_RuntimeError, "Failed to create initial swapchain");
        return NULL;
    }

    // ---- Command pool ----
    {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = ctx->graphics_family;
        vkCreateCommandPool(ctx->device, &pool_info, nullptr, &ctx->command_pool);
    }

    // ---- Per-frame resources ----
    {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = ctx->command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkAllocateCommandBuffers(ctx->device, &alloc_info,
                                     &ctx->frames[i].command_buffer);
            vkAllocateCommandBuffers(ctx->device, &alloc_info,
                                     &ctx->frames[i].post_rt_command_buffer);
            vkCreateSemaphore(ctx->device, &sem_info, nullptr,
                              &ctx->frames[i].image_available);
            vkCreateSemaphore(ctx->device, &sem_info, nullptr,
                              &ctx->frames[i].render_finished);
            vkCreateFence(ctx->device, &fence_info, nullptr,
                          &ctx->frames[i].in_flight_fence);
        }
    }

    ctx->current_frame = 0;
    ctx->in_frame = false;
    ctx->in_render_pass = false;
    ctx->descriptor_dirty = false;

    memset(ctx->bound_vertex_buffers, 0, sizeof(ctx->bound_vertex_buffers));
    memset(ctx->bound_ubo_buffers, 0, sizeof(ctx->bound_ubo_buffers));
    memset(ctx->bound_textures, 0, sizeof(ctx->bound_textures));

    // ---- Descriptors ----
    create_descriptor_infrastructure(ctx);

    // ---- Dummy resources (used by flush_descriptors to fill any
    // descriptor binding the active shader's layout declares but the
    // current draw hasn't bound, so validation is happy).
    create_dummy_resources(ctx);

    // ---- Timeline semaphore for Metal RT sync ----
    ctx->rt_timeline_semaphore = VK_NULL_HANDLE;
    ctx->rt_sync_value = 0;
    ctx->rt_sync_pending = false;
#ifdef __APPLE__
    if (ctx->metal_objects_supported) {
        // Create a timeline semaphore with export intent for MTLSharedEvent.
        VkSemaphoreTypeCreateInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline_info.initialValue = 0;

        VkExportMetalObjectCreateInfoEXT export_create{};
        export_create.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT;
        export_create.exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_SHARED_EVENT_BIT_EXT;

        // Chain: sem_info -> timeline_info -> export_create
        timeline_info.pNext = &export_create;

        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        sem_info.pNext = &timeline_info;

        VkResult sem_result = vkCreateSemaphore(ctx->device, &sem_info, nullptr,
                                                &ctx->rt_timeline_semaphore);
        if (sem_result != VK_SUCCESS) {
            fprintf(stderr, "[VK] Timeline semaphore creation failed: %d\n", sem_result);
            ctx->rt_timeline_semaphore = VK_NULL_HANDLE;
        } else {
            fprintf(stderr, "[VK] Timeline semaphore created: %p\n",
                    (void*)ctx->rt_timeline_semaphore);
        }
    }
#endif

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(ctx));
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_destroy_context(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;

    vkDeviceWaitIdle(ctx->device);

    // Flush any queued resources from in-flight frames now that the GPU
    // is fully idle.
    flush_all_pending_deletes(ctx);

    // Release every device child that may still be represented by a stale
    // Python integer handle.  Pipelines must precede their layouts/render
    // passes; buffers and images must precede the VMA allocator.
    destroy_raytracing_resources(ctx);
    for (VulkanPipeline* pipeline : ctx->live_pipelines) {
        if (pipeline->pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(ctx->device, pipeline->pipeline, nullptr);
        delete pipeline;
    }
    ctx->live_pipelines.clear();

    for (TextureWithDevice* tex : ctx->live_textures) {
        if (tex->sampler != VK_NULL_HANDLE)
            vkDestroySampler(tex->device, tex->sampler, nullptr);
        if (tex->image_view != VK_NULL_HANDLE)
            vkDestroyImageView(tex->device, tex->image_view, nullptr);
        if (tex->image != VK_NULL_HANDLE)
            vmaDestroyImage(tex->allocator, tex->image, tex->allocation);
        delete tex;
    }
    ctx->live_textures.clear();

    for (BufferWithAllocator* buffer : ctx->live_buffers) {
        if (buffer->base.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(buffer->allocator, buffer->base.buffer,
                             buffer->base.allocation);
        delete buffer;
    }
    ctx->live_buffers.clear();

    // OIT owns views into the current depth target and must be gone before
    // cleanup_swapchain destroys that target.
    destroy_oit_resources_internal(ctx);

    for (OffscreenFramebuffer* framebuffer : ctx->live_offscreens)
        destroy_offscreen_resource_internal(ctx, framebuffer);
    ctx->live_offscreens.clear();
    ctx->active_offscreen = nullptr;

    // Dummy descriptors used by flush_descriptors fill-ins.
    destroy_dummy_resources(ctx);

    // Descriptor infrastructure
    if (ctx->pipeline_cache != VK_NULL_HANDLE)
        vkDestroyPipelineCache(ctx->device, ctx->pipeline_cache, nullptr);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (ctx->descriptor_pools[i] != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pools[i], nullptr);
    }
    if (ctx->shared_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(ctx->device, ctx->shared_pipeline_layout, nullptr);
    if (ctx->descriptor_set_layout_textures != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(ctx->device, ctx->descriptor_set_layout_textures, nullptr);
    if (ctx->descriptor_set_layout_ubo != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(ctx->device, ctx->descriptor_set_layout_ubo, nullptr);

    // Per-frame resources
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(ctx->device, ctx->frames[i].image_available, nullptr);
        vkDestroySemaphore(ctx->device, ctx->frames[i].render_finished, nullptr);
        vkDestroyFence(ctx->device, ctx->frames[i].in_flight_fence, nullptr);
    }
    if (ctx->rt_timeline_semaphore != VK_NULL_HANDLE)
        vkDestroySemaphore(ctx->device, ctx->rt_timeline_semaphore, nullptr);
    vkDestroyCommandPool(ctx->device, ctx->command_pool, nullptr);

    // Swapchain and associated resources
    cleanup_swapchain(ctx);
    if (ctx->owns_surface && ctx->surface != VK_NULL_HANDLE)
        vkDestroySurfaceKHR(ctx->instance, ctx->surface, nullptr);
    ctx->surface = VK_NULL_HANDLE;
    ctx->owns_surface = false;

    // Inactive windows retain their own swapchain/depth/render-pass objects.
    // They must all be gone before the allocator, device, or instance.
    for (auto& entry : ctx->window_swapchains)
        destroy_window_swapchain(ctx, entry.second);
    ctx->window_swapchains.clear();

    // Non-owning wrapper handed out by vulkan_get_depth_texture; its
    // image/view/sampler belong to depth_image and are gone already.
    delete ctx->depth_texture_wrapper;
    ctx->depth_texture_wrapper = nullptr;

    // VMA allocator
    vmaDestroyAllocator(ctx->allocator);

    VkInstance instance = ctx->instance;
#ifndef NDEBUG
    VkDebugUtilsMessengerEXT messenger = ctx->debug_messenger;
#endif

    vkDestroyDevice(ctx->device, nullptr);

#ifndef NDEBUG
    auto destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy_fn && messenger != VK_NULL_HANDLE)
        destroy_fn(instance, messenger, nullptr);
#endif

    if (ctx->owns_instance)
        vkDestroyInstance(instance, nullptr);

    delete ctx;
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_surface_for_window(PyObject* self, PyObject* args,
                                        PyObject* kwds)
{
    Py_ssize_t ctx_ptr, window_handle;
    Py_ssize_t surface_handle = 0;
    int owns_surface_arg = -1;
    int width = 0, height = 0;
    static const char* kwlist[] = {"context", "window_handle", "surface",
                                   "owns_surface", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn|niii"), (char**)kwlist,
            &ctx_ptr, &window_handle, &surface_handle, &owns_surface_arg,
            &width, &height))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    const std::uintptr_t target_window =
        static_cast<std::uintptr_t>(window_handle);
    const uint32_t target_width = width > 0
        ? static_cast<uint32_t>(width) : 0;
    const uint32_t target_height = height > 0
        ? static_cast<uint32_t>(height) : 0;

    if (ctx->in_frame) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Cannot switch Vulkan surfaces during a frame");
        return NULL;
    }

    if (target_window == ctx->current_window_handle) {
        if (target_width && target_height &&
                (target_width != ctx->width || target_height != ctx->height)) {
            VkFormat old_format = ctx->swapchain_format;
            if (!recreate_active_swapchain(ctx, target_width, target_height)) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Failed to resize target window swapchain");
                return NULL;
            }
            return PyBool_FromLong(old_format != ctx->swapchain_format);
        }
        Py_RETURN_FALSE;
    }

    auto cached = ctx->window_swapchains.find(target_window);
    const bool target_is_cached = cached != ctx->window_swapchains.end();
    VkSurfaceKHR new_surface = reinterpret_cast<VkSurfaceKHR>(surface_handle);
    bool owns_new_surface = owns_surface_arg >= 0
        ? owns_surface_arg != 0 : surface_handle == 0;

#if defined(_WIN32)
    if (!target_is_cached && new_surface == VK_NULL_HANDLE) {
        VkWin32SurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        info.hinstance = GetModuleHandle(nullptr);
        info.hwnd = reinterpret_cast<HWND>(window_handle);
        if (vkCreateWin32SurfaceKHR(ctx->instance, &info, nullptr,
                                   &new_surface) != VK_SUCCESS) {
            PyErr_SetString(PyExc_RuntimeError,
                            "vkCreateWin32SurfaceKHR failed for target window");
            return NULL;
        }
        owns_new_surface = true;
    }
#else
    (void)window_handle;
    if (!target_is_cached && new_surface == VK_NULL_HANDLE) {
        PyErr_SetString(PyExc_ValueError,
                        "A platform VkSurfaceKHR is required for this window");
        return NULL;
    }
#endif

    if (!target_is_cached) {
        VkBool32 supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(
            ctx->physical_device, ctx->present_family, new_surface, &supported);
        if (!supported) {
            if (owns_new_surface && new_surface != VK_NULL_HANDLE)
                vkDestroySurfaceKHR(ctx->instance, new_surface, nullptr);
            PyErr_SetString(PyExc_RuntimeError,
                            "Vulkan present queue does not support target surface");
            return NULL;
        }
    }

    // OIT is disabled for Vulkan's deterministic transparency path.  If a
    // caller explicitly created it, retire it once before changing depth
    // targets; ordinary window switches remain wait-free.
    if (ctx->oit_render_pass != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx->device);
        flush_all_pending_deletes(ctx);
        destroy_oit_resources_internal(ctx);
    }

    const std::uintptr_t previous_window = ctx->current_window_handle;
    const VkFormat previous_format = ctx->swapchain_format;
    auto saved = ctx->window_swapchains.emplace(
        previous_window, WindowSwapchainState{});
    if (!saved.second) {
        if (!target_is_cached && owns_new_surface && new_surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(ctx->instance, new_surface, nullptr);
        PyErr_SetString(PyExc_RuntimeError,
                        "Active Vulkan window already exists in surface cache");
        return NULL;
    }
    take_window_swapchain(ctx, saved.first->second);

    if (target_is_cached) {
        // emplace above may have rehashed the unordered_map, so resolve the
        // target again instead of retaining an invalidated iterator.
        auto target_state = ctx->window_swapchains.find(target_window);
        restore_window_swapchain(ctx, target_state->second);
        ctx->window_swapchains.erase(target_state);
    } else {
        ctx->surface = new_surface;
        ctx->owns_surface = owns_new_surface;
        const uint32_t new_width = target_width ? target_width : 1;
        const uint32_t new_height = target_height ? target_height : 1;
        if (!create_swapchain(ctx, new_width, new_height)) {
            if (ctx->owns_surface && ctx->surface != VK_NULL_HANDLE)
                vkDestroySurfaceKHR(ctx->instance, ctx->surface, nullptr);
            ctx->surface = VK_NULL_HANDLE;
            ctx->owns_surface = false;
            restore_window_swapchain(ctx, saved.first->second);
            ctx->window_swapchains.erase(saved.first);
            ctx->current_window_handle = previous_window;
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to create swapchain for target window");
            return NULL;
        }
    }
    ctx->current_window_handle = target_window;

    if (target_is_cached && target_width && target_height &&
            (ctx->swapchain == VK_NULL_HANDLE ||
             target_width != ctx->width || target_height != ctx->height)) {
        if (!recreate_active_swapchain(ctx, target_width, target_height)) {
            // Keep Python/native window selection coherent on failure. The
            // target surface remains reusable (with no swapchain), while the
            // previously active window is restored before the exception is
            // returned to Python.
            WindowSwapchainState failed_target{};
            take_window_swapchain(ctx, failed_target);
            auto previous_state = ctx->window_swapchains.find(previous_window);
            restore_window_swapchain(ctx, previous_state->second);
            ctx->window_swapchains.erase(previous_state);
            ctx->window_swapchains.emplace(
                target_window, std::move(failed_target));
            ctx->current_window_handle = previous_window;
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to resize cached target swapchain");
            return NULL;
        }
    }

    return PyBool_FromLong(previous_format != ctx->swapchain_format);
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_release_window(PyObject* self, PyObject* args,
                                PyObject* kwds)
{
    Py_ssize_t ctx_ptr, window_handle;
    static const char* kwlist[] = {"context", "window_handle", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist,
            &ctx_ptr, &window_handle))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    const std::uintptr_t window = static_cast<std::uintptr_t>(window_handle);
    if (window == ctx->current_window_handle) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Cannot release the active Vulkan window");
        return NULL;
    }

    auto found = ctx->window_swapchains.find(window);
    if (found == ctx->window_swapchains.end())
        Py_RETURN_NONE;

    vkDeviceWaitIdle(ctx->device);
    flush_all_pending_deletes(ctx);
    destroy_window_swapchain(ctx, found->second);
    ctx->window_swapchains.erase(found);
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_resize_drawable(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    int width, height;
    static const char* kwlist[] = {"context", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nii"), (char**)kwlist, &ctx_ptr, &width, &height))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);

    // Minimized/hidden Qt windows can transiently report an empty drawable.
    // Keep the existing swapchain until a real size arrives instead of
    // passing wrapped negative/zero dimensions into Vulkan.
    if (width <= 0 || height <= 0)
        Py_RETURN_NONE;

    if (static_cast<uint32_t>(width) == ctx->width &&
            static_cast<uint32_t>(height) == ctx->height)
        Py_RETURN_NONE;

    if (!recreate_active_swapchain(ctx, static_cast<uint32_t>(width),
                                   static_cast<uint32_t>(height))) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to recreate swapchain");
        return NULL;
    }

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_device_name(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;

    return PyUnicode_FromString(ctx->device_name.c_str());
}

extern "C"
PyObject* vulkan_max_3d_texture_size(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;
    return PyLong_FromUnsignedLong(ctx->max_3d_texture_size);
}

extern "C"
PyObject* vulkan_supports_ray_tracing(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;
    if (ctx->raytracing_supported) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

extern "C"
PyObject* vulkan_supports_metal_objects(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;
    if (ctx->metal_objects_supported) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

// ---------------------------------------------------------------------------
//  Submit the current command buffer and wait for GPU completion.
//  Used before Metal RT dispatch so the depth buffer is ready.
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_flush_gpu(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;

    if (!ctx->in_frame) Py_RETURN_NONE;

    FrameData& frame = ctx->frames[ctx->current_frame];

    // End any active render pass (queues owned pass objects for deletion).
    end_active_render_pass(ctx);

    vkEndCommandBuffer(frame.command_buffer);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.command_buffer;

    // This may be the frame's first submit, in which case it (not
    // end_frame's submit) must wait on the swapchain acquire semaphore —
    // the commands recorded so far already target the acquired image.
    VkPipelineStageFlags acquire_wait_stage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (ctx->image_acquired_unconsumed) {
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &frame.image_available;
        submit.pWaitDstStageMask = &acquire_wait_stage;
        ctx->image_acquired_unconsumed = false;
    }

    // If we have a timeline semaphore for Metal RT sync, signal it so
    // the Metal dispatch can wait on the GPU without a CPU stall.
    VkTimelineSemaphoreSubmitInfo timeline_submit{};
    uint64_t signal_value = 0;
    if (ctx->rt_timeline_semaphore != VK_NULL_HANDLE) {
        ctx->rt_sync_value += 1;
        signal_value = ctx->rt_sync_value;

        timeline_submit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit.signalSemaphoreValueCount = 1;
        timeline_submit.pSignalSemaphoreValues = &signal_value;
        submit.pNext = &timeline_submit;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &ctx->rt_timeline_semaphore;
    }

    vkQueueSubmit(ctx->graphics_queue, 1, &submit, VK_NULL_HANDLE);

    // Swap to the post-RT command buffer for subsequent Vulkan work in this
    // frame. The submitted buffer is now pending on the GPU; same-queue
    // ordering guarantees it completes before end_frame's fenced submit,
    // so begin_frame's fence wait covers both.
    std::swap(frame.command_buffer, frame.post_rt_command_buffer);
    vkResetCommandBuffer(frame.command_buffer, 0);
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(frame.command_buffer, &begin_info);

    // Re-apply dynamic state on the fresh command buffer.
    if (ctx->has_viewport) {
        vkCmdSetViewport(frame.command_buffer, 0, 1, &ctx->pending_viewport);
        vkCmdSetScissor(frame.command_buffer, 0, 1, &ctx->pending_scissor);
    }

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_begin_frame(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;
    VK_TRACE("begin_frame current_frame=%u fence=%p sem=%p cmd=%p swap=%p",
             ctx->current_frame,
             (void*)ctx->frames[ctx->current_frame].in_flight_fence,
             (void*)ctx->frames[ctx->current_frame].image_available,
             (void*)ctx->frames[ctx->current_frame].command_buffer,
             (void*)ctx->swapchain);

    FrameData& frame = ctx->frames[ctx->current_frame];

    // Do not block Qt's GUI thread waiting for a frame slot.  This context
    // is shared by the main and auxiliary native windows, and a delayed
    // child-window presentation must not prevent the main event loop from
    // committing the very drawable it is waiting for.  Python treats False
    // as a skipped frame and retries on a later event-loop turn.
    VK_TRACE("  waitForFences");
    VkResult fence_status = vkGetFenceStatus(
        ctx->device, frame.in_flight_fence);
    if (fence_status == VK_NOT_READY)
        Py_RETURN_FALSE;
    if (fence_status != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to query Vulkan frame fence");
        return NULL;
    }

    // Now safe to actually free anything queued during the previous use of
    // this frame slot — the GPU just finished executing those commands.
    flush_pending_deletes(ctx, ctx->pending_deletes[ctx->current_frame]);

    // Skip swapchain image acquisition when rendering offscreen for image
    // capture — there's no swapchain image to acquire and end_frame won't
    // present. The frame still uses the per-frame command buffer + fence so
    // synchronisation with deferred deletes stays correct.
    if (ctx->active_offscreen == nullptr) {
        // A previous resize or recreate may have failed and left the
        // context without a swapchain; acquiring from VK_NULL_HANDLE
        // would crash, so retry creation here.
        if (ctx->swapchain == VK_NULL_HANDLE) {
            if (!create_swapchain(ctx, ctx->width, ctx->height)) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Swapchain unavailable and recreation failed");
                return NULL;
            }
        }

        VK_TRACE("  acquireNextImage");
        VkResult result;
        // A Qt child window may be exposed before its platform drawable is
        // ready, especially while a dock is opening on macOS. Acquisition is
        // deliberately non-blocking: Qt/Core Animation may need another
        // event-loop turn to make the drawable available, and Python will
        // retry the skipped frame on the next update.
        constexpr uint64_t acquire_timeout_ns = 0;
        Py_BEGIN_ALLOW_THREADS
        result = vkAcquireNextImageKHR(
            ctx->device, ctx->swapchain, acquire_timeout_ns,
            frame.image_available, VK_NULL_HANDLE,
            &ctx->current_image_index);
        Py_END_ALLOW_THREADS
        VK_TRACE("  acquireNextImage -> %d image_index=%u", (int)result, ctx->current_image_index);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // Swapchain is no longer valid; recreate it. Drain deferred
            // deletes first — queued framebuffers reference the swapchain
            // and depth views cleanup_swapchain is about to destroy.
            vkDeviceWaitIdle(ctx->device);
            flush_all_pending_deletes(ctx);
            cleanup_swapchain(ctx);
            if (!create_swapchain(ctx, ctx->width, ctx->height)) {
                PyErr_SetString(PyExc_RuntimeError,
                                "Failed to recreate swapchain after OUT_OF_DATE");
                return NULL;
            }
            // Retry acquisition with the new swapchain.
            result = vkAcquireNextImageKHR(
                ctx->device, ctx->swapchain, acquire_timeout_ns,
                frame.image_available, VK_NULL_HANDLE,
                &ctx->current_image_index);
        }

        if (result == VK_TIMEOUT || result == VK_NOT_READY)
            Py_RETURN_FALSE;

        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to acquire swapchain image");
            return NULL;
        }

        // The first submit of this frame must wait on image_available.
        ctx->image_acquired_unconsumed = true;
    }

    VK_TRACE("  resetFences");
    vkResetFences(ctx->device, 1, &frame.in_flight_fence);

    // Reset this frame's descriptor pool so flush_descriptors can hand out
    // fresh sets for every draw without invalidating bindings recorded
    // earlier in the cmd buffer.
    VK_TRACE("  resetDescriptorPool");
    vkResetDescriptorPool(ctx->device, ctx->descriptor_pools[ctx->current_frame], 0);

    // Reset both command buffers — post_rt may have been used by flush_gpu.
    VK_TRACE("  resetCommandBuffer");
    vkResetCommandBuffer(frame.command_buffer, 0);
    vkResetCommandBuffer(frame.post_rt_command_buffer, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_TRACE("  beginCommandBuffer");
    vkBeginCommandBuffer(frame.command_buffer, &begin_info);

    ctx->in_frame = true;
    ctx->in_render_pass = false;

    // Re-apply cached viewport/scissor now that the cmd buffer is recording.
    if (ctx->has_viewport) {
        vkCmdSetViewport(frame.command_buffer, 0, 1, &ctx->pending_viewport);
        vkCmdSetScissor(frame.command_buffer, 0, 1, &ctx->pending_scissor);
    }

    VK_TRACE("  begin_frame done");

    Py_RETURN_TRUE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_end_frame(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;
    VK_TRACE("end_frame");

    FrameData& frame = ctx->frames[ctx->current_frame];

    // End any active render pass (queues owned pass objects for deletion).
    end_active_render_pass(ctx);

    vkEndCommandBuffer(frame.command_buffer);

    // Submit the command buffer. When rendering offscreen for image
    // capture there's no swapchain image to wait on or present, so skip
    // the image-available semaphore and the present call. The fence is
    // still signalled so begin_frame's wait keeps frame slots in sync.
    bool offscreen = (ctx->active_offscreen != nullptr);

    // flush_gpu may already have consumed the acquire semaphore as the
    // frame's first submit; waiting on a binary semaphore twice is invalid.
    bool wait_acquire = !offscreen && ctx->image_acquired_unconsumed;

    // Pick the per-image render-finished semaphore so present's wait
    // is on a semaphore tied to the image being presented, not the
    // frame slot. Reusing a per-frame-slot semaphore for present trips
    // VUID-vkQueueSubmit-pSignalSemaphores-00067 when acquireNextImage
    // returns the same image index multiple times before the previous
    // present's semaphore has been released.
    VkSemaphore render_finished_sem = VK_NULL_HANDLE;
    if (!offscreen) {
        if (ctx->current_image_index >= ctx->render_finished_per_image.size()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Swapchain image index out of range in end_frame");
            return NULL;
        }
        render_finished_sem = ctx->render_finished_per_image[ctx->current_image_index];
    }

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    if (wait_acquire) {
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &frame.image_available;
        submit_info.pWaitDstStageMask = &wait_stage;
    }
    if (!offscreen) {
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_finished_sem;
    }
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &frame.command_buffer;

    // If Metal RT signalled completion, this submit must wait for it.
    VkTimelineSemaphoreSubmitInfo timeline_wait{};
    VkSemaphore wait_sems[2];
    VkPipelineStageFlags wait_stages[2];
    uint64_t wait_values[2] = {0, 0};

    if (ctx->rt_sync_pending && ctx->rt_timeline_semaphore != VK_NULL_HANDLE) {
        // Build combined wait array: image_available + rt_timeline.
        uint32_t wait_count = 0;
        if (wait_acquire) {
            wait_sems[0] = frame.image_available;
            wait_stages[0] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            wait_values[0] = 0;  // binary semaphore, value ignored
            wait_count = 1;
        }
        wait_sems[wait_count] = ctx->rt_timeline_semaphore;
        wait_stages[wait_count] = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        wait_values[wait_count] = ctx->rt_sync_value;
        wait_count++;

        timeline_wait.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_wait.waitSemaphoreValueCount = wait_count;
        timeline_wait.pWaitSemaphoreValues = wait_values;

        submit_info.pNext = &timeline_wait;
        submit_info.waitSemaphoreCount = wait_count;
        submit_info.pWaitSemaphores = wait_sems;
        submit_info.pWaitDstStageMask = wait_stages;

        ctx->rt_sync_pending = false;
    }

    vkQueueSubmit(ctx->graphics_queue, 1, &submit_info, frame.in_flight_fence);
    ctx->image_acquired_unconsumed = false;

    if (offscreen) {
        // Wait for the offscreen draw to complete so vulkan_read_framebuffer
        // can safely copy out of the color image.
        Py_BEGIN_ALLOW_THREADS
        vkWaitForFences(ctx->device, 1, &frame.in_flight_fence, VK_TRUE, UINT64_MAX);
        Py_END_ALLOW_THREADS
        ctx->in_frame = false;
        ctx->current_frame = (ctx->current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Py_RETURN_NONE;
    }

    // Present, waiting on the same per-image semaphore the submit signalled.
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_finished_sem;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &ctx->swapchain;
    present_info.pImageIndices = &ctx->current_image_index;

    VkResult result;
    Py_BEGIN_ALLOW_THREADS
    result = vkQueuePresentKHR(ctx->present_queue, &present_info);
    Py_END_ALLOW_THREADS

    // The frame is over regardless of how present went; keep the slot
    // bookkeeping consistent before any error return so the next
    // begin_frame waits on the right fence.
    ctx->in_frame = false;
    ctx->current_frame = (ctx->current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        vkDeviceWaitIdle(ctx->device);
        // See vulkan_resize_drawable: deferred deletes must drain before
        // the views their framebuffers reference are destroyed.
        flush_all_pending_deletes(ctx);
        cleanup_swapchain(ctx);
        if (!create_swapchain(ctx, ctx->width, ctx->height)) {
            // begin_frame retries creation when swapchain is null; report
            // the failure instead of silently continuing with no swapchain.
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to recreate swapchain after present");
            return NULL;
        }
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to present swapchain image");
        return NULL;
    }

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_begin_render_pass(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    float clear_r, clear_g, clear_b, clear_a;
    int clear_depth;
    int load_color = 0;
    int load_depth = 0;
    static const char* kwlist[] = {"context", "clear_r", "clear_g", "clear_b", "clear_a",
                                   "clear_depth", "load_color", "load_depth", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nffffi|ii"), (char**)kwlist,
            &ctx_ptr, &clear_r, &clear_g, &clear_b, &clear_a,
            &clear_depth, &load_color, &load_depth))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("begin_render_pass clear=(%.2f,%.2f,%.2f,%.2f) cd=%d lc=%d ld=%d", clear_r, clear_g, clear_b, clear_a, clear_depth, load_color, load_depth);
    reset_descriptor_bindings(ctx);
    FrameData& frame = ctx->frames[ctx->current_frame];

    // Image-capture path: render to the offscreen framebuffer. A render
    // pass honoring the requested load/clear ops is built per call (and
    // defer-deleted), mirroring the swapchain path — capture frames with
    // transparency or overlays resume with load_color/load_depth, and
    // always clearing here would wipe the opaque scene before compositing.
    OffscreenFramebuffer* off = ctx->active_offscreen;
    if (off != nullptr) {
        end_active_render_pass(ctx);

        // Until the first pass has rendered, the attachments are in
        // UNDEFINED layout with garbage contents; demote loads to clears.
        if (off->color_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            load_color = 0;
            load_depth = 0;
        }
        bool depth_load = (load_depth || !clear_depth);

        VkAttachmentDescription attachments[2] = {};
        attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;  // offscreen color format
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = load_color ? VK_ATTACHMENT_LOAD_OP_LOAD
                                           : VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // Every offscreen pass ends in TRANSFER_SRC_OPTIMAL (for readback),
        // so that's also where a loaded image starts.
        attachments[0].initialLayout = load_color ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                                  : VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = depth_load ? VK_ATTACHMENT_LOAD_OP_LOAD
                                           : VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = depth_load
            ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            : VK_IMAGE_LAYOUT_UNDEFINED;
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

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = 2;
        rp_info.pAttachments = attachments;
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 1;
        rp_info.pDependencies = &dependency;

        VkRenderPass render_pass;
        if (vkCreateRenderPass(ctx->device, &rp_info, nullptr, &render_pass) != VK_SUCCESS) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create offscreen render pass");
            return NULL;
        }

        VkClearValue clear_values[2];
        clear_values[0].color = {{clear_r, clear_g, clear_b, off->alpha ? clear_a : 1.0f}};
        clear_values[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp_begin{};
        rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_begin.renderPass = render_pass;
        // off->framebuffer was created against the load-op-compatible
        // pre-built pass; load/store ops don't affect compatibility.
        rp_begin.framebuffer = off->framebuffer;
        rp_begin.renderArea.offset = {0, 0};
        rp_begin.renderArea.extent = {off->width, off->height};
        rp_begin.clearValueCount = 2;
        rp_begin.pClearValues = clear_values;

        vkCmdBeginRenderPass(frame.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
        ctx->in_render_pass = true;
        off->color_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // finalLayout

        RenderPassState state;
        state.render_pass = render_pass;
        state.framebuffer = off->framebuffer;
        state.owns_render_pass = true;
        state.owns_framebuffer = false;
        ctx->render_pass_stack.push_back(state);
        Py_RETURN_NONE;
    }

    // End any previously active render pass.
    end_active_render_pass(ctx);

    // Loading color from a swapchain image that has never been rendered to
    // since (re)creation would declare initialLayout PRESENT_SRC while the
    // image is actually UNDEFINED with garbage contents. Clear instead.
    if (load_color &&
        (ctx->current_image_index >= ctx->swapchain_image_used.size() ||
         !ctx->swapchain_image_used[ctx->current_image_index]))
        load_color = 0;

    // Create a render pass with the requested load/store ops.
    VkAttachmentDescription attachments[2] = {};

    // Color
    attachments[0].format = ctx->swapchain_format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = load_color ? VK_ATTACHMENT_LOAD_OP_LOAD
                                       : VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = load_color ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                              : VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Depth
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = (load_depth || !clear_depth) ? VK_ATTACHMENT_LOAD_OP_LOAD
                                                         : VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // When loading depth, initialLayout MUST match the image's actual
    // layout. Compute passes (RTAO) may leave depth in SHADER_READ_ONLY,
    // so honour ctx->depth_image.current_layout when set.
    if (load_depth || !clear_depth) {
        VkImageLayout actual = ctx->depth_image.current_layout;
        attachments[1].initialLayout = (actual != VK_IMAGE_LAYOUT_UNDEFINED)
            ? actual
            : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    } else {
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
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

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 1;
    rp_info.pDependencies = &dependency;

    VkRenderPass render_pass;
    VkResult vr = vkCreateRenderPass(ctx->device, &rp_info, nullptr, &render_pass);
    if (vr != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create render pass");
        return NULL;
    }

    // Create a temporary framebuffer for this render pass.
    VkImageView fb_attachments[] = {
        ctx->swapchain_image_views[ctx->current_image_index],
        ctx->depth_image.image_view
    };

    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = fb_attachments;
    fb_info.width = ctx->swapchain_extent.width;
    fb_info.height = ctx->swapchain_extent.height;
    fb_info.layers = 1;

    VkFramebuffer framebuffer;
    vr = vkCreateFramebuffer(ctx->device, &fb_info, nullptr, &framebuffer);
    if (vr != VK_SUCCESS) {
        vkDestroyRenderPass(ctx->device, render_pass, nullptr);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create framebuffer");
        return NULL;
    }

    // Begin the render pass.
    VkClearValue clear_values[2];
    clear_values[0].color = {{clear_r, clear_g, clear_b, clear_a}};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea.offset = {0, 0};
    rp_begin.renderArea.extent = ctx->swapchain_extent;
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;

    vkCmdBeginRenderPass(frame.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    ctx->in_render_pass = true;
    if (ctx->current_image_index < ctx->swapchain_image_used.size())
        ctx->swapchain_image_used[ctx->current_image_index] = 1;
    // The depth attachment's load op transitions the image into subpass
    // layout, and the finalLayout above governs where it lands after
    // vkCmdEndRenderPass. Track that so the next begin_render_pass can
    // pick the right initialLayout (e.g. after RTAO leaves it sampled).
    ctx->depth_image.current_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Track the render pass for cleanup.
    RenderPassState state;
    state.render_pass = render_pass;
    state.framebuffer = framebuffer;
    state.owns_render_pass = true;
    state.owns_framebuffer = true;
    ctx->render_pass_stack.push_back(state);

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_begin_depth_pass(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr, tex_ptr;
    int width, height;
    static const char* kwlist[] = {"context", "depth_texture", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nnii"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &width, &height))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanTexture* tex = reinterpret_cast<VulkanTexture*>(tex_ptr);
    VK_TRACE("begin_depth_pass tex=%p %dx%d", (void*)tex, width, height);
    reset_descriptor_bindings(ctx);
    FrameData& frame = ctx->frames[ctx->current_frame];

    // End any active render pass.
    end_active_render_pass(ctx);

    // Transition the depth image to attachment-optimal if needed.
    if (tex->current_layout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
        tex->current_layout != VK_IMAGE_LAYOUT_UNDEFINED) {
        transition_image_layout(frame.command_buffer, tex->image,
                                tex->current_layout,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // Create an image view for the target texture if we do not already have one.
    VkImageView depth_view = tex->image_view;

    // Create a temporary framebuffer targeting the depth texture.
    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = ctx->depth_only_render_pass;
    fb_info.attachmentCount = 1;
    fb_info.pAttachments = &depth_view;
    fb_info.width = static_cast<uint32_t>(width);
    fb_info.height = static_cast<uint32_t>(height);
    fb_info.layers = 1;

    VkFramebuffer framebuffer;
    VkResult vr = vkCreateFramebuffer(ctx->device, &fb_info, nullptr, &framebuffer);
    if (vr != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create depth framebuffer");
        return NULL;
    }

    VkClearValue clear_value;
    clear_value.depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = ctx->depth_only_render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea.offset = {0, 0};
    rp_begin.renderArea.extent = {static_cast<uint32_t>(width),
                                  static_cast<uint32_t>(height)};
    rp_begin.clearValueCount = 1;
    rp_begin.pClearValues = &clear_value;

    vkCmdBeginRenderPass(frame.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    ctx->in_render_pass = true;
    // The depth-only render pass's finalLayout is SHADER_READ_ONLY_OPTIMAL,
    // so that's where the image lands when the pass ends. Track the
    // post-pass layout — current_layout is only consulted outside the pass,
    // and recording ATTACHMENT_OPTIMAL here would leave later barriers with
    // a stale oldLayout after the first shadow pass.
    tex->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    RenderPassState state;
    state.render_pass = VK_NULL_HANDLE;  // shared pass, do not destroy
    state.framebuffer = framebuffer;
    state.owns_render_pass = false;
    state.owns_framebuffer = true;
    ctx->render_pass_stack.push_back(state);

    // Set viewport and scissor to match the depth texture dimensions.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_end_render_pass(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;
    VK_TRACE("end_render_pass in_pass=%d", ctx->in_render_pass ? 1 : 0);

    if (!ctx->in_render_pass) {
        Py_RETURN_NONE;
    }

    FrameData& frame = ctx->frames[ctx->current_frame];
    vkCmdEndRenderPass(frame.command_buffer);
    ctx->in_render_pass = false;

    // Queue per-pass framebuffer/render-pass for destruction after the
    // current frame's GPU work completes; the command buffer still
    // references them at this point. end_render_pass always runs inside
    // a frame (in_frame=true), so deferred_delete_slot returns
    // current_frame -- but go through the helper for consistency.
    if (!ctx->render_pass_stack.empty()) {
        RenderPassState& state = ctx->render_pass_stack.back();
        PendingDeletes& pd = ctx->pending_deletes[deferred_delete_slot(ctx)];
        if (state.owns_framebuffer && state.framebuffer != VK_NULL_HANDLE)
            pd.framebuffers.push_back(state.framebuffer);
        if (state.owns_render_pass && state.render_pass != VK_NULL_HANDLE)
            pd.render_passes.push_back(state.render_pass);
        ctx->render_pass_stack.pop_back();
    }

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_begin_color_pass(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr, tex_ptr;
    int width, height;
    int do_clear = 1;
    static const char* kwlist[] = {"context", "color_texture", "width", "height",
                                   "do_clear", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nnii|i"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &width, &height, &do_clear))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanTexture* tex = reinterpret_cast<VulkanTexture*>(tex_ptr);
    VK_TRACE("begin_color_pass tex=%p %dx%d clear=%d", (void*)tex, width, height, do_clear);
    reset_descriptor_bindings(ctx);
    FrameData& frame = ctx->frames[ctx->current_frame];

    // End any active render pass.
    end_active_render_pass(ctx);

    // Transition image if needed.
    if (tex->current_layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
        tex->current_layout != VK_IMAGE_LAYOUT_UNDEFINED) {
        transition_image_layout(frame.command_buffer, tex->image,
                                tex->current_layout,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // Create a color-only render pass.
    VkAttachmentDescription color_att{};
    color_att.format = tex->format;
    color_att.samples = VK_SAMPLE_COUNT_1_BIT;
    color_att.loadOp = do_clear ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                : VK_ATTACHMENT_LOAD_OP_LOAD;
    color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_att.initialLayout = do_clear ? VK_IMAGE_LAYOUT_UNDEFINED
                                       : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_att.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 1;
    rp_info.pAttachments = &color_att;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 1;
    rp_info.pDependencies = &dependency;

    VkRenderPass render_pass;
    VkResult vr = vkCreateRenderPass(ctx->device, &rp_info, nullptr, &render_pass);
    if (vr != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create color render pass");
        return NULL;
    }

    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = 1;
    fb_info.pAttachments = &tex->image_view;
    fb_info.width = static_cast<uint32_t>(width);
    fb_info.height = static_cast<uint32_t>(height);
    fb_info.layers = 1;

    VkFramebuffer framebuffer;
    vr = vkCreateFramebuffer(ctx->device, &fb_info, nullptr, &framebuffer);
    if (vr != VK_SUCCESS) {
        vkDestroyRenderPass(ctx->device, render_pass, nullptr);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create color framebuffer");
        return NULL;
    }

    VkClearValue clear_value;
    clear_value.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea.offset = {0, 0};
    rp_begin.renderArea.extent = {static_cast<uint32_t>(width),
                                  static_cast<uint32_t>(height)};
    rp_begin.clearValueCount = 1;
    rp_begin.pClearValues = &clear_value;

    vkCmdBeginRenderPass(frame.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    ctx->in_render_pass = true;
    tex->current_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    RenderPassState state;
    state.render_pass = render_pass;
    state.framebuffer = framebuffer;
    state.owns_render_pass = true;
    state.owns_framebuffer = true;
    ctx->render_pass_stack.push_back(state);

    // Set viewport and scissor.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Highlight-outline mask pass: render into a swapchain-format color
//  texture while depth-testing against the live scene depth buffer. The
//  attachment formats (swapchain color + D32 depth) deliberately match
//  the main render pass so the ordinary scene pipelines are render-pass
//  compatible and can draw the highlighted geometry into the mask.
// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_begin_mask_pass(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr, tex_ptr;
    int width, height;
    static const char* kwlist[] = {"context", "color_texture", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nnii"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &width, &height))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanTexture* tex = reinterpret_cast<VulkanTexture*>(tex_ptr);
    VK_TRACE("begin_mask_pass tex=%p %dx%d", (void*)tex, width, height);
    if (!ctx->in_frame) Py_RETURN_NONE;

    // The mask shares the scene depth image; sizes must agree or the
    // framebuffer would be invalid.
    if (static_cast<uint32_t>(width) != ctx->swapchain_extent.width ||
        static_cast<uint32_t>(height) != ctx->swapchain_extent.height) {
        PyErr_Format(PyExc_RuntimeError,
                     "Mask size %dx%d does not match render target %ux%u",
                     width, height,
                     ctx->swapchain_extent.width, ctx->swapchain_extent.height);
        return NULL;
    }

    reset_descriptor_bindings(ctx);
    FrameData& frame = ctx->frames[ctx->current_frame];

    end_active_render_pass(ctx);

    VkAttachmentDescription attachments[2] = {};
    // Mask color: clear to transparent black, end samplable for the
    // outline composite quad.
    attachments[0].format = tex->format;  // == ctx->swapchain_format
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Scene depth: loaded read-only in effect (callers draw with depth
    // writes disabled), so hidden parts of highlighted geometry are
    // excluded from the mask without copying the depth buffer.
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    {
        VkImageLayout actual = ctx->depth_image.current_layout;
        attachments[1].initialLayout = (actual != VK_IMAGE_LAYOUT_UNDEFINED)
            ? actual
            : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
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

    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    // Make the mask writes visible to the composite quad's fragment reads.
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 2;
    rp_info.pDependencies = deps;

    VkRenderPass render_pass;
    if (vkCreateRenderPass(ctx->device, &rp_info, nullptr, &render_pass) != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create mask render pass");
        return NULL;
    }

    VkImageView fb_attachments[2] = { tex->image_view, ctx->depth_image.image_view };
    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = fb_attachments;
    fb_info.width = static_cast<uint32_t>(width);
    fb_info.height = static_cast<uint32_t>(height);
    fb_info.layers = 1;

    VkFramebuffer framebuffer;
    if (vkCreateFramebuffer(ctx->device, &fb_info, nullptr, &framebuffer) != VK_SUCCESS) {
        vkDestroyRenderPass(ctx->device, render_pass, nullptr);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create mask framebuffer");
        return NULL;
    }

    VkClearValue clear_values[2];
    clear_values[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clear_values[1].depthStencil = {1.0f, 0};  // unused (depth is loaded)

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea.offset = {0, 0};
    rp_begin.renderArea.extent = {static_cast<uint32_t>(width),
                                  static_cast<uint32_t>(height)};
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;

    vkCmdBeginRenderPass(frame.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
    ctx->in_render_pass = true;
    tex->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // finalLayout

    RenderPassState state;
    state.render_pass = render_pass;
    state.framebuffer = framebuffer;
    state.owns_render_pass = true;
    state.owns_framebuffer = true;
    ctx->render_pass_stack.push_back(state);

    // Re-apply viewport/scissor on the new render pass.
    if (ctx->has_viewport) {
        vkCmdSetViewport(frame.command_buffer, 0, 1, &ctx->pending_viewport);
        vkCmdSetScissor(frame.command_buffer, 0, 1, &ctx->pending_scissor);
    }

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_viewport(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    double x, y, w, h;
    static const char* kwlist[] = {"context", "x", "y", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ndddd"), (char**)kwlist,
            &ctx_ptr, &x, &y, &w, &h))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VK_TRACE("set_viewport (%.0f,%.0f) %.0fx%.0f in_frame=%d", x, y, w, h, ctx->in_frame ? 1 : 0);

    ctx->pending_viewport.x = static_cast<float>(x);
    ctx->pending_viewport.y = static_cast<float>(y);
    ctx->pending_viewport.width = static_cast<float>(w);
    ctx->pending_viewport.height = static_cast<float>(h);
    ctx->pending_viewport.minDepth = 0.0f;
    ctx->pending_viewport.maxDepth = 1.0f;

    ctx->pending_scissor.offset = {static_cast<int32_t>(x), static_cast<int32_t>(y)};
    ctx->pending_scissor.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    ctx->has_viewport = true;

    // If a frame is currently recording, emit immediately so subsequent draws
    // use the new state. Otherwise the values are applied in begin_frame.
    if (ctx->in_frame) {
        VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;
        vkCmdSetViewport(cmd, 0, 1, &ctx->pending_viewport);
        vkCmdSetScissor(cmd, 0, 1, &ctx->pending_scissor);
    }

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_cull_mode(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr;
    int mode;  // 0 = none, 1 = front, 2 = back
    static const char* kwlist[] = {"context", "mode", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ni"), (char**)kwlist, &ctx_ptr, &mode))
        return NULL;

    // Vulkan 1.0 does not support dynamic cull mode.  We store the requested
    // mode so that pipeline creation can bake it in.  If the device supports
    // VK_EXT_extended_dynamic_state we could use vkCmdSetCullMode, but for
    // now we simply record the value for later pipeline selection.
    //
    // This matches the Metal backend's approach of recording state that is
    // applied when the next draw call selects a pipeline.
    (void)mode;

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Copy the scene depth buffer into a sampleable depth texture. Used by
//  the silhouette pass: the live depth image is an attachment of the main
//  render pass and can't simultaneously be sampled, so the edge-detect
//  quad reads a copy instead. Ends any active render pass; the caller
//  resumes with load_color/load_depth afterwards.
// ---------------------------------------------------------------------------
extern "C"
PyObject* vulkan_copy_depth_texture(PyObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t ctx_ptr, tex_ptr;
    static const char* kwlist[] = {"context", "texture", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist, &ctx_ptr, &tex_ptr))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanTexture* tex = reinterpret_cast<VulkanTexture*>(tex_ptr);
    if (!ctx->in_frame) Py_RETURN_NONE;

    if (tex->width != ctx->depth_image.width ||
        tex->height != ctx->depth_image.height ||
        tex->format != ctx->depth_image.format) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Depth copy texture does not match scene depth buffer");
        return NULL;
    }

    end_active_render_pass(ctx);
    VkCommandBuffer cmd = ctx->frames[ctx->current_frame].command_buffer;

    VkImageLayout depth_layout = ctx->depth_image.current_layout;
    transition_image_layout(cmd, ctx->depth_image.image, depth_layout,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_ASPECT_DEPTH_BIT);
    transition_image_layout(cmd, tex->image, tex->current_layout,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_ASPECT_DEPTH_BIT);

    VkImageCopy region{};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.dstSubresource.layerCount = 1;
    region.extent = { ctx->depth_image.width, ctx->depth_image.height, 1 };
    vkCmdCopyImage(cmd,
                   ctx->depth_image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   tex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region);

    transition_image_layout(cmd, ctx->depth_image.image,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_ASPECT_DEPTH_BIT);
    ctx->depth_image.current_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    transition_image_layout(cmd, tex->image,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_ASPECT_DEPTH_BIT);
    tex->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_get_depth_texture(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;

    if (ctx->depth_image.image == VK_NULL_HANDLE) {
        Py_RETURN_NONE;
    }

    // Wrap the main depth image in a VulkanTexture so it can be bound like
    // any other texture (for example, for SSAO reads). One wrapper per
    // context, refreshed on every call — this is typically called per
    // frame, so allocating a new wrapper each time would leak steadily.
    // The wrapper must never be passed to destroy_texture; it owns nothing.
    VulkanTexture* tex = ctx->depth_texture_wrapper;
    if (tex == nullptr) {
        tex = new VulkanTexture();
        ctx->depth_texture_wrapper = tex;
    }
    tex->image = ctx->depth_image.image;
    tex->allocation = VK_NULL_HANDLE;  // not owned by this wrapper
    tex->image_view = ctx->depth_image.image_view;
    tex->format = VK_FORMAT_D32_SFLOAT;
    tex->width = ctx->depth_image.width;
    tex->height = ctx->depth_image.height;
    tex->current_layout = ctx->depth_image.current_layout;

    // Create a sampler with nearest filtering for raw depth reads.
    if (ctx->depth_image.sampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samp_info{};
        samp_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samp_info.magFilter = VK_FILTER_NEAREST;
        samp_info.minFilter = VK_FILTER_NEAREST;
        samp_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samp_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samp_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samp_info.maxAnisotropy = 1.0f;
        samp_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        samp_info.unnormalizedCoordinates = VK_FALSE;
        samp_info.compareEnable = VK_FALSE;
        samp_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

        vkCreateSampler(ctx->device, &samp_info, nullptr, &ctx->depth_image.sampler);
    }

    tex->sampler = ctx->depth_image.sampler;

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(tex));
}
