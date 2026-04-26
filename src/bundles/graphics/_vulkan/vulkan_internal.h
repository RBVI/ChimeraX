/*
 * Internal header shared between Vulkan .cpp files.
 * Defines state structs so all files can access their fields.
 *
 * Uses vk-bootstrap for device/swapchain setup and VMA for memory.
 * Both are included as single-header/few-file vendored libraries.
 */

#ifndef CHIMERAX_VULKAN_INTERNAL_H
#define CHIMERAX_VULKAN_INTERNAL_H

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <vector>
#include <unordered_map>
#include <string>

// Maximum frames in flight for double-buffering.
static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

// Maximum descriptor set uniform buffer bindings.
static constexpr uint32_t MAX_UBO_BINDINGS = 5;
static constexpr uint32_t MAX_TEXTURE_BINDINGS = 8;

// ---------------------------------------------------------------------------
//  Per-frame synchronisation objects and command buffer.
// ---------------------------------------------------------------------------
struct FrameData {
    VkCommandBuffer command_buffer;
    VkSemaphore     image_available;
    VkSemaphore     render_finished;
    VkFence         in_flight_fence;
};

// ---------------------------------------------------------------------------
//  Vulkan buffer wrapper (vertex, index, or uniform data).
// ---------------------------------------------------------------------------
struct VulkanBuffer {
    VkBuffer      buffer;
    VmaAllocation allocation;
    VkDeviceSize  size;
};

// ---------------------------------------------------------------------------
//  Vulkan texture wrapper (VkImage + VkImageView + VkSampler).
// ---------------------------------------------------------------------------
struct VulkanTexture {
    VkImage       image;
    VmaAllocation allocation;
    VkImageView   image_view;
    VkSampler     sampler;
    VkFormat      format;
    uint32_t      width;
    uint32_t      height;
    VkImageLayout current_layout;
};

// ---------------------------------------------------------------------------
//  Vulkan pipeline wrapper.
// ---------------------------------------------------------------------------
struct VulkanPipeline {
    VkPipeline       pipeline;
    VkPipelineLayout layout;
};

// ---------------------------------------------------------------------------
//  Render pass state for begin_pass / end_pass tracking.
// ---------------------------------------------------------------------------
struct RenderPassState {
    VkRenderPass  render_pass;
    VkFramebuffer framebuffer;
    bool          owns_render_pass;   // true if we created it (not the swapchain pass)
    bool          owns_framebuffer;
};

// ---------------------------------------------------------------------------
//  Main Vulkan context — one per graphics window.
// ---------------------------------------------------------------------------
struct VulkanContext {
    // Core objects
    VkInstance       instance;
    VkPhysicalDevice physical_device;
    VkDevice         device;
    VkQueue          graphics_queue;
    VkQueue          present_queue;
    uint32_t         graphics_family;
    uint32_t         present_family;
    VkSurfaceKHR     surface;
    VmaAllocator     allocator;

    // Swapchain
    VkSwapchainKHR           swapchain;
    VkFormat                 swapchain_format;
    VkExtent2D               swapchain_extent;
    std::vector<VkImage>     swapchain_images;
    std::vector<VkImageView> swapchain_image_views;

    // Depth buffer (for the main render pass)
    VulkanTexture depth_image;

    // Main render pass and framebuffers (one per swapchain image)
    VkRenderPass              main_render_pass;
    VkRenderPass              depth_only_render_pass;
    std::vector<VkFramebuffer> swapchain_framebuffers;

    // Command pool and per-frame data
    VkCommandPool command_pool;
    FrameData     frames[MAX_FRAMES_IN_FLIGHT];
    uint32_t      current_frame;        // index into frames[]
    uint32_t      current_image_index;  // swapchain image acquired this frame
    bool          in_frame;
    bool          in_render_pass;

    // Active render pass stack (for nested begin_pass/end_pass)
    std::vector<RenderPassState> render_pass_stack;

    // Pipeline layout shared across all pipelines
    // (uniform buffers at set 0, textures at set 1)
    VkDescriptorSetLayout descriptor_set_layout_ubo;
    VkDescriptorSetLayout descriptor_set_layout_textures;
    VkPipelineLayout      shared_pipeline_layout;

    // Descriptor pool and sets (one set per frame in flight)
    VkDescriptorPool descriptor_pool;

    // Pipeline cache for faster pipeline creation
    VkPipelineCache pipeline_cache;

    // Per-frame descriptor sets
    VkDescriptorSet ubo_descriptor_sets[MAX_FRAMES_IN_FLIGHT];
    VkDescriptorSet texture_descriptor_sets[MAX_FRAMES_IN_FLIGHT];

    // Currently bound state (for deferred binding)
    VulkanBuffer*   bound_vertex_buffers[16];  // indexed by binding
    VulkanBuffer*   bound_ubo_buffers[MAX_UBO_BINDINGS];
    VulkanTexture*  bound_textures[MAX_TEXTURE_BINDINGS];
    bool            descriptor_dirty;

    // Size
    uint32_t width;
    uint32_t height;

    // Device name
    std::string device_name;

    // Debug messenger (validation layers in debug builds)
#ifndef NDEBUG
    VkDebugUtilsMessengerEXT debug_messenger;
#endif
};

// ---------------------------------------------------------------------------
//  Helper to extract context pointer from Python keyword args.
// ---------------------------------------------------------------------------
static inline VulkanContext* get_vulkan_context(PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    static const char* kwlist[] = {"context", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("l"), (char**)kwlist, &ctx_ptr))
        return nullptr;
    return reinterpret_cast<VulkanContext*>(ctx_ptr);
}

// ---------------------------------------------------------------------------
//  Forward declarations for internal helpers (defined in vulkan_context.cpp).
// ---------------------------------------------------------------------------
bool create_swapchain(VulkanContext* ctx, uint32_t width, uint32_t height);
void cleanup_swapchain(VulkanContext* ctx);
void create_depth_resources(VulkanContext* ctx);
void create_main_render_pass(VulkanContext* ctx);
void create_swapchain_framebuffers(VulkanContext* ctx);
void create_descriptor_infrastructure(VulkanContext* ctx);
VkShaderModule create_shader_module(VulkanContext* ctx, const uint32_t* code, size_t size);
void transition_image_layout(VkCommandBuffer cmd, VkImage image,
                             VkImageLayout old_layout, VkImageLayout new_layout,
                             VkImageAspectFlags aspect);

#endif // CHIMERAX_VULKAN_INTERNAL_H
