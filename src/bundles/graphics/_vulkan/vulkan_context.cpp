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
    img_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;
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
        depth_att.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 0;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 0;
        subpass.pColorAttachments = nullptr;
        subpass.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.attachmentCount = 1;
        rp_info.pAttachments = &depth_att;
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 1;
        rp_info.pDependencies = &dependency;

        vkCreateRenderPass(ctx->device, &rp_info, nullptr, &ctx->depth_only_render_pass);
    }
}

// ---------------------------------------------------------------------------

bool create_swapchain(VulkanContext* ctx, uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder builder{ctx->physical_device, ctx->device, ctx->surface};

    auto swap_ret = builder
        .set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
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

    if (ctx->swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(ctx->device, ctx->swapchain, nullptr);
        ctx->swapchain = VK_NULL_HANDLE;
    }
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
    // ---- Set 0: Uniform buffers (bindings 0..4) ----
    {
        VkDescriptorSetLayoutBinding bindings[MAX_UBO_BINDINGS];
        for (uint32_t i = 0; i < MAX_UBO_BINDINGS; ++i) {
            bindings[i] = {};
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                     VK_SHADER_STAGE_FRAGMENT_BIT;
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

    // ---- Descriptor pool ----
    {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              MAX_UBO_BINDINGS * MAX_FRAMES_IN_FLIGHT },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              MAX_TEXTURE_BINDINGS * MAX_FRAMES_IN_FLIGHT },
        };

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = MAX_FRAMES_IN_FLIGHT * 2;  // 2 sets per frame
        pool_info.poolSizeCount = 2;
        pool_info.pPoolSizes = pool_sizes;

        vkCreateDescriptorPool(ctx->device, &pool_info, nullptr, &ctx->descriptor_pool);
    }

    // ---- Allocate per-frame descriptor sets ----
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkDescriptorSetLayout layouts_ubo[] = { ctx->descriptor_set_layout_ubo };
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = ctx->descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = layouts_ubo;
        vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->ubo_descriptor_sets[i]);

        VkDescriptorSetLayout layouts_tex[] = { ctx->descriptor_set_layout_textures };
        alloc_info.pSetLayouts = layouts_tex;
        vkAllocateDescriptorSets(ctx->device, &alloc_info, &ctx->texture_descriptor_sets[i]);
    }

    // ---- Pipeline cache ----
    {
        VkPipelineCacheCreateInfo cache_info{};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        vkCreatePipelineCache(ctx->device, &cache_info, nullptr, &ctx->pipeline_cache);
    }
}

// ===========================================================================
//  Public API — extern "C" functions matching vulkan_api.h
// ===========================================================================

extern "C"
PyObject* vulkan_create_context(PyObject* self, PyObject* args, PyObject* kwds)
{
    long surface_handle;
    static const char* kwlist[] = {"surface_handle", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("l"), (char**)kwlist, &surface_handle))
        return NULL;

    // ---- Instance ----
    vkb::InstanceBuilder inst_builder;
    inst_builder.set_app_name("ChimeraX")
                .require_api_version(1, 0, 0);

#ifndef NDEBUG
    inst_builder.request_validation_layers()
                .use_default_debug_messenger();
#endif

    auto inst_ret = inst_builder.build();
    if (!inst_ret) {
        PyErr_Format(PyExc_RuntimeError, "Failed to create Vulkan instance: %s",
                     inst_ret.error().message().c_str());
        return NULL;
    }
    vkb::Instance vkb_inst = inst_ret.value();

    VkSurfaceKHR surface = reinterpret_cast<VkSurfaceKHR>(surface_handle);

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
            vkb::destroy_instance(vkb_inst);
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to select Vulkan physical device: %s",
                         phys_ret.error().message().c_str());
            return NULL;
        }
    }

    vkb::PhysicalDevice vkb_phys = phys_ret.value();

    // ---- Logical device ----
    vkb::DeviceBuilder dev_builder{vkb_phys};
    auto dev_ret = dev_builder.build();
    if (!dev_ret) {
        vkb::destroy_instance(vkb_inst);
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to create Vulkan logical device: %s",
                     dev_ret.error().message().c_str());
        return NULL;
    }
    vkb::Device vkb_dev = dev_ret.value();

    // ---- Allocate and populate context ----
    VulkanContext* ctx = new VulkanContext();
    memset(ctx, 0, sizeof(VulkanContext));

    ctx->instance = vkb_inst.instance;
    ctx->surface = surface;
    ctx->physical_device = vkb_phys.physical_device;
    ctx->device = vkb_dev.device;

    auto gq = vkb_dev.get_queue(vkb::QueueType::graphics);
    auto pq = vkb_dev.get_queue(vkb::QueueType::present);
    if (!gq || !pq) {
        delete ctx;
        vkb::destroy_device(vkb_dev);
        vkb::destroy_instance(vkb_inst);
        PyErr_SetString(PyExc_RuntimeError, "Failed to get Vulkan queues");
        return NULL;
    }
    ctx->graphics_queue = gq.value();
    ctx->present_queue = pq.value();
    ctx->graphics_family = vkb_dev.get_queue_index(vkb::QueueType::graphics).value();
    ctx->present_family = vkb_dev.get_queue_index(vkb::QueueType::present).value();

    // Store the device name.
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &props);
    ctx->device_name = props.deviceName;

#ifndef NDEBUG
    ctx->debug_messenger = vkb_inst.debug_messenger;
#endif

    // ---- VMA allocator ----
    {
        VmaAllocatorCreateInfo alloc_info{};
        alloc_info.physicalDevice = ctx->physical_device;
        alloc_info.device = ctx->device;
        alloc_info.instance = ctx->instance;

        VkResult vr = vmaCreateAllocator(&alloc_info, &ctx->allocator);
        if (vr != VK_SUCCESS) {
            delete ctx;
            vkb::destroy_device(vkb_dev);
            vkb::destroy_instance(vkb_inst);
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
        vkb::destroy_instance(vkb_inst);
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

    return PyLong_FromLong(reinterpret_cast<long>(ctx));
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_destroy_context(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;

    vkDeviceWaitIdle(ctx->device);

    // Descriptor infrastructure
    if (ctx->pipeline_cache != VK_NULL_HANDLE)
        vkDestroyPipelineCache(ctx->device, ctx->pipeline_cache, nullptr);
    if (ctx->descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, nullptr);
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
    vkDestroyCommandPool(ctx->device, ctx->command_pool, nullptr);

    // Swapchain and associated resources
    cleanup_swapchain(ctx);

    // VMA allocator
    vmaDestroyAllocator(ctx->allocator);

    // Surface is owned by the caller (Python/Qt side); do not destroy it here.

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

    vkDestroyInstance(instance, nullptr);

    delete ctx;
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_resize_drawable(PyObject* self, PyObject* args, PyObject* kwds)
{
    long ctx_ptr;
    int width, height;
    static const char* kwlist[] = {"context", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lii"), (char**)kwlist, &ctx_ptr, &width, &height))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);

    vkDeviceWaitIdle(ctx->device);
    cleanup_swapchain(ctx);

    if (!create_swapchain(ctx, static_cast<uint32_t>(width),
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

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_begin_frame(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;

    FrameData& frame = ctx->frames[ctx->current_frame];

    // Wait for this frame's previous work to finish.
    vkWaitForFences(ctx->device, 1, &frame.in_flight_fence, VK_TRUE, UINT64_MAX);

    // Acquire the next swapchain image.
    VkResult result = vkAcquireNextImageKHR(
        ctx->device, ctx->swapchain, UINT64_MAX,
        frame.image_available, VK_NULL_HANDLE,
        &ctx->current_image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        // Swapchain is no longer valid; recreate it.
        vkDeviceWaitIdle(ctx->device);
        cleanup_swapchain(ctx);
        if (!create_swapchain(ctx, ctx->width, ctx->height)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to recreate swapchain after OUT_OF_DATE");
            return NULL;
        }
        // Retry acquisition with the new swapchain.
        result = vkAcquireNextImageKHR(
            ctx->device, ctx->swapchain, UINT64_MAX,
            frame.image_available, VK_NULL_HANDLE,
            &ctx->current_image_index);
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to acquire swapchain image");
        return NULL;
    }

    vkResetFences(ctx->device, 1, &frame.in_flight_fence);

    // Reset and begin the command buffer.
    vkResetCommandBuffer(frame.command_buffer, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(frame.command_buffer, &begin_info);

    ctx->in_frame = true;
    ctx->in_render_pass = false;

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_end_frame(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;

    FrameData& frame = ctx->frames[ctx->current_frame];

    // End any active render pass.
    if (ctx->in_render_pass) {
        vkCmdEndRenderPass(frame.command_buffer);
        ctx->in_render_pass = false;
        ctx->render_pass_stack.clear();
    }

    vkEndCommandBuffer(frame.command_buffer);

    // Submit the command buffer.
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &frame.image_available;
    submit_info.pWaitDstStageMask = &wait_stage;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &frame.command_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &frame.render_finished;

    vkQueueSubmit(ctx->graphics_queue, 1, &submit_info, frame.in_flight_fence);

    // Present.
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &frame.render_finished;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &ctx->swapchain;
    present_info.pImageIndices = &ctx->current_image_index;

    VkResult result = vkQueuePresentKHR(ctx->present_queue, &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        vkDeviceWaitIdle(ctx->device);
        cleanup_swapchain(ctx);
        create_swapchain(ctx, ctx->width, ctx->height);
    } else if (result != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to present swapchain image");
        return NULL;
    }

    ctx->in_frame = false;
    ctx->current_frame = (ctx->current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_begin_render_pass(PyObject* self, PyObject* args, PyObject* kwds)
{
    long ctx_ptr;
    float clear_r, clear_g, clear_b, clear_a;
    int clear_depth;
    int load_color = 0;
    int load_depth = 0;
    static const char* kwlist[] = {"context", "clear_r", "clear_g", "clear_b", "clear_a",
                                   "clear_depth", "load_color", "load_depth", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lffffi|ii"), (char**)kwlist,
            &ctx_ptr, &clear_r, &clear_g, &clear_b, &clear_a,
            &clear_depth, &load_color, &load_depth))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    FrameData& frame = ctx->frames[ctx->current_frame];

    // End any previously active render pass.
    if (ctx->in_render_pass) {
        vkCmdEndRenderPass(frame.command_buffer);
        ctx->in_render_pass = false;
    }

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
    attachments[1].initialLayout = (load_depth || !clear_depth)
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
    long ctx_ptr, tex_ptr;
    int width, height;
    static const char* kwlist[] = {"context", "depth_texture", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llii"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &width, &height))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanTexture* tex = reinterpret_cast<VulkanTexture*>(tex_ptr);
    FrameData& frame = ctx->frames[ctx->current_frame];

    // End any active render pass.
    if (ctx->in_render_pass) {
        vkCmdEndRenderPass(frame.command_buffer);
        ctx->in_render_pass = false;
    }

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
    tex->current_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

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

    if (!ctx->in_render_pass) {
        Py_RETURN_NONE;
    }

    FrameData& frame = ctx->frames[ctx->current_frame];
    vkCmdEndRenderPass(frame.command_buffer);
    ctx->in_render_pass = false;

    // Clean up the most recent render pass state.
    if (!ctx->render_pass_stack.empty()) {
        RenderPassState& state = ctx->render_pass_stack.back();
        // Queue these for deferred destruction (they are still referenced
        // by the in-flight command buffer).  For simplicity, destroy them
        // after the device is idle at the end of frame.  A production
        // renderer would use per-frame deletion queues.
        if (state.owns_framebuffer && state.framebuffer != VK_NULL_HANDLE)
            vkDestroyFramebuffer(ctx->device, state.framebuffer, nullptr);
        if (state.owns_render_pass && state.render_pass != VK_NULL_HANDLE)
            vkDestroyRenderPass(ctx->device, state.render_pass, nullptr);
        ctx->render_pass_stack.pop_back();
    }

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_begin_color_pass(PyObject* self, PyObject* args, PyObject* kwds)
{
    long ctx_ptr, tex_ptr;
    int width, height;
    int do_clear = 1;
    static const char* kwlist[] = {"context", "color_texture", "width", "height",
                                   "do_clear", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llii|i"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &width, &height, &do_clear))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    VulkanTexture* tex = reinterpret_cast<VulkanTexture*>(tex_ptr);
    FrameData& frame = ctx->frames[ctx->current_frame];

    // End any active render pass.
    if (ctx->in_render_pass) {
        vkCmdEndRenderPass(frame.command_buffer);
        ctx->in_render_pass = false;
    }

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

extern "C"
PyObject* vulkan_set_viewport(PyObject* self, PyObject* args, PyObject* kwds)
{
    long ctx_ptr;
    double x, y, w, h;
    static const char* kwlist[] = {"context", "x", "y", "width", "height", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ldddd"), (char**)kwlist,
            &ctx_ptr, &x, &y, &w, &h))
        return NULL;

    VulkanContext* ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);
    FrameData& frame = ctx->frames[ctx->current_frame];

    VkViewport viewport{};
    viewport.x = static_cast<float>(x);
    viewport.y = static_cast<float>(y);
    viewport.width = static_cast<float>(w);
    viewport.height = static_cast<float>(h);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {static_cast<int32_t>(x), static_cast<int32_t>(y)};
    scissor.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------

extern "C"
PyObject* vulkan_set_cull_mode(PyObject* self, PyObject* args, PyObject* kwds)
{
    long ctx_ptr;
    int mode;  // 0 = none, 1 = front, 2 = back
    static const char* kwlist[] = {"context", "mode", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("li"), (char**)kwlist, &ctx_ptr, &mode))
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

extern "C"
PyObject* vulkan_get_depth_texture(PyObject* self, PyObject* args, PyObject* kwds)
{
    VulkanContext* ctx = get_vulkan_context(args, kwds);
    if (!ctx) return NULL;

    if (ctx->depth_image.image == VK_NULL_HANDLE) {
        Py_RETURN_NONE;
    }

    // Wrap the main depth image in a VulkanTexture so it can be bound like
    // any other texture (for example, for SSAO reads).
    VulkanTexture* tex = new VulkanTexture();
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

    return PyLong_FromLong(reinterpret_cast<long>(tex));
}
