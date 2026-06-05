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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnullability-completeness"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-private-field"
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop

#include <vector>
#include <unordered_map>
#include <string>

#include <cstdio>
#include <cstdarg>

// Trace helper used during Vulkan backend bring-up. Writes to vulkan_trace.log
// in the current working directory and to stderr, flushing on every line so
// we can read the last call before a hard crash.
inline void vk_trace(const char* fmt, ...) {
    static FILE* f = nullptr;
    if (!f) {
        f = fopen("vulkan_trace.log", "w");
    }
    va_list ap;
    va_start(ap, fmt);
    if (f) {
        va_list ap2;
        va_copy(ap2, ap);
        vfprintf(f, fmt, ap2);
        fputc('\n', f);
        fflush(f);
        va_end(ap2);
    }
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}
// Compile out trace logging in release builds (NDEBUG defined). The
// no-op form still consumes its arguments inside sizeof so the
// compiler typechecks the call sites; nothing is actually emitted.
#ifdef NDEBUG
#  define VK_TRACE(...) ((void)sizeof(0, ##__VA_ARGS__))
#else
#  define VK_TRACE(...) vk_trace(__VA_ARGS__)
#endif

// Maximum frames in flight for double-buffering.
static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

// Set 0 buffer bindings (mix of UBO and SSBO; the layout below picks
// per-binding type to match the GLSL shader declarations):
//   0,1,5,6 = uniform blocks (UBO)  (6 = volume raycast params)
//   2,3,4   = readonly storage blocks (SSBO) for variable-length data
//             (multishadow matrices, AO offsets, AO neighbors).
static constexpr uint32_t MAX_UBO_BINDINGS = 7;
static constexpr uint32_t MAX_TEXTURE_BINDINGS = 8;

inline VkDescriptorType set0_descriptor_type(uint32_t binding) {
    return (binding == 2 || binding == 3 || binding == 4)
        ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
}

// ---------------------------------------------------------------------------
//  Per-frame synchronisation objects and command buffer.
// ---------------------------------------------------------------------------
struct FrameData {
    VkCommandBuffer command_buffer;
    VkCommandBuffer post_rt_command_buffer;  // used after flush_gpu for post-RT work
    VkSemaphore     image_available;
    VkSemaphore     render_finished;
    VkFence         in_flight_fence;
};

// Forward declarations so wrappers can carry a back-pointer to the context
// (used by destroy paths to defer frees onto the current frame's queue).
struct VulkanContext;

// ---------------------------------------------------------------------------
//  Vulkan buffer wrapper (vertex, index, or uniform data).
// ---------------------------------------------------------------------------
struct VulkanBuffer {
    VkBuffer      buffer;
    VmaAllocation allocation;
    VkDeviceSize  size;
};

// Allocator-aware buffer wrapper used by vulkan_buffer.cpp. Lives here so
// the deferred-deletion queue can free it without a back-reference to the
// .cpp file. Carries the owning context so destroy_buffer can route the
// release through the right per-frame deletion queue.
struct BufferWithAllocator {
    VulkanBuffer    base;
    VmaAllocator    allocator;
    VulkanContext*  ctx;
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
    uint32_t      depth;        // 1 for 1D / 2D textures, >1 for 3D
    uint32_t      dimension;    // 1, 2, or 3
    VkImageLayout current_layout;
};

// Device-aware texture wrapper used by vulkan_texture.cpp.
struct TextureWithDevice : VulkanTexture {
    VkDevice       device;
    VmaAllocator   allocator;
    VulkanContext* ctx;
};

// Per-frame deferred-deletion queue. Populated as objects are released
// during recording, drained at begin_frame after the corresponding fence
// has signalled (i.e. the GPU is finished with that frame's commands).
struct PendingDeletes {
    std::vector<VkFramebuffer>          framebuffers;
    std::vector<VkRenderPass>           render_passes;
    std::vector<BufferWithAllocator*>   buffers;
    std::vector<TextureWithDevice*>     textures;
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
//  Offscreen framebuffer used for image capture. Owns its color/depth
//  attachments and a render pass. While push_framebuffer makes one active,
//  the swapchain code path is bypassed: begin_frame skips vkAcquireNextImage
//  and end_frame skips vkQueuePresentKHR. The color image is always
//  R8G8B8A8_UNORM with TRANSFER_SRC_BIT so vulkan_read_framebuffer can copy
//  it into a staging buffer for read-back into a numpy array.
// ---------------------------------------------------------------------------
struct OffscreenFramebuffer {
    VkImage       color_image;
    VmaAllocation color_alloc;
    VkImageView   color_view;
    VkImage       depth_image;
    VmaAllocation depth_alloc;
    VkImageView   depth_view;
    VkFramebuffer framebuffer;
    VkRenderPass  render_pass;
    uint32_t      width;
    uint32_t      height;
    bool          alpha;          // false -> opaque output (alpha forced to 1)
    VkImageLayout color_layout;   // tracked across begin/end + readback
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
    // One render-finished semaphore per swapchain image. The semaphore
    // passed to vkQueuePresentKHR is held by the presentation engine
    // until the image is re-acquired, so reusing it from a frame slot
    // (FrameData::render_finished) trips
    // VUID-vkQueueSubmit-pSignalSemaphores-00067 when the same image
    // index is acquired multiple times in a row. Index this vector by
    // current_image_index so each present uses a semaphore tied to the
    // image being presented.
    std::vector<VkSemaphore> render_finished_per_image;

    // Depth buffer (for the main render pass)
    VulkanTexture depth_image;

    // Main render pass and framebuffers (one per swapchain image)
    VkRenderPass              main_render_pass;
    VkRenderPass              depth_only_render_pass;
    std::vector<VkFramebuffer> swapchain_framebuffers;

    // OIT (Weighted Blended Order-Independent Transparency).
    // Two color attachments: accum (RGBA16F, additive) and revealage
    // (R8, multiplicative). Shares the main depth buffer (read-only).
    VkRenderPass  oit_render_pass;
    VkFramebuffer oit_framebuffer;
    VkImage       oit_accum_image;
    VmaAllocation oit_accum_alloc;
    VkImageView   oit_accum_view;
    VkImage       oit_reveal_image;
    VmaAllocation oit_reveal_alloc;
    VkImageView   oit_reveal_view;
    VulkanTexture oit_accum_texture;   // samplable handle for composite
    VulkanTexture oit_reveal_texture;  // samplable handle for composite
    uint32_t      oit_width, oit_height;

    // Command pool and per-frame data
    VkCommandPool command_pool;
    FrameData     frames[MAX_FRAMES_IN_FLIGHT];
    uint32_t      current_frame;        // index into frames[]
    uint32_t      current_image_index;  // swapchain image acquired this frame
    bool          in_frame;
    bool          in_render_pass;

    // Active render pass stack (for nested begin_pass/end_pass)
    std::vector<RenderPassState> render_pass_stack;

    // While non-null, the next frame renders to this offscreen target
    // instead of the swapchain. Set by vulkan_set_active_offscreen,
    // cleared by passing 0. begin_frame/begin_render_pass/end_frame each
    // branch on this so the existing swapchain path is unaffected when
    // there's no active capture.
    OffscreenFramebuffer* active_offscreen;

    // Dummy resources written to descriptor-set bindings the shader
    // statically references but the user hasn't bound for this draw.
    // Vulkan validation requires every layout-declared binding the
    // bound pipeline references to have a valid descriptor written; if
    // a sampler/buffer slot is left unwritten, the runtime fires
    // VUID-vkCmdDrawIndexed-None-08114 even when the shader's runtime
    // path never samples from it. flush_descriptors fills any unbound
    // slot with the matching dummy below.
    struct {
        // 2D R8G8B8A8 color (samplers at set 1 bindings 0, 7)
        VkImage       tex2d_image;
        VmaAllocation tex2d_alloc;
        VkImageView   tex2d_view;
        // 3D R8G8B8A8 color (sampler3D at bindings 1, 5)
        VkImage       tex3d_image;
        VmaAllocation tex3d_alloc;
        VkImageView   tex3d_view;
        // Cube R8G8B8A8 color (samplerCube at binding 2)
        VkImage       texcube_image;
        VmaAllocation texcube_alloc;
        VkImageView   texcube_view;
        // 1D R8G8B8A8 color (sampler1D at binding 6)
        VkImage       tex1d_image;
        VmaAllocation tex1d_alloc;
        VkImageView   tex1d_view;
        // D32 depth with compare sampler (sampler2DShadow at bindings 3, 4)
        VkImage       depth_image;
        VmaAllocation depth_alloc;
        VkImageView   depth_view;
        // Two samplers: regular for color textures, shadow-compare for depth
        VkSampler     color_sampler;
        VkSampler     shadow_sampler;
        // Buffers for set-0 UBO / SSBO bindings
        VkBuffer      ubo_buffer;
        VmaAllocation ubo_alloc;
        VkBuffer      ssbo_buffer;
        VmaAllocation ssbo_alloc;
    } dummy;

    // Pipeline layout shared across all pipelines
    // (uniform buffers at set 0, textures at set 1)
    VkDescriptorSetLayout descriptor_set_layout_ubo;
    VkDescriptorSetLayout descriptor_set_layout_textures;
    VkPipelineLayout      shared_pipeline_layout;

    // One descriptor pool per frame in flight. flush_descriptors allocates
    // a fresh pair of sets out of the current frame's pool every time it
    // runs; the pool is reset at the start of each frame. This avoids
    // updating a descriptor set that was already bound to the in-flight
    // command buffer (Vulkan disallows that without UPDATE_AFTER_BIND).
    VkDescriptorPool descriptor_pools[MAX_FRAMES_IN_FLIGHT];

    // Per-frame deferred-deletion queues for resources released during
    // recording (framebuffers from end_render_pass, buffers/textures from
    // Python destroy calls). Drained at begin_frame after the fence wait,
    // so the GPU is guaranteed done before any vkDestroy* runs.
    PendingDeletes pending_deletes[MAX_FRAMES_IN_FLIGHT];

    // Pipeline cache for faster pipeline creation
    VkPipelineCache pipeline_cache;

    // Currently bound state (for deferred binding)
    VulkanBuffer*   bound_vertex_buffers[16];  // indexed by binding
    VulkanBuffer*   bound_ubo_buffers[MAX_UBO_BINDINGS];
    VulkanTexture*  bound_textures[MAX_TEXTURE_BINDINGS];
    bool            descriptor_dirty;

    // Size
    uint32_t width;
    uint32_t height;

    // Pending viewport/scissor state. set_viewport() can be called outside a
    // frame; the commands must only be emitted while the cmd buffer is in
    // the recording state, so we cache the values and apply them in
    // begin_frame and at the start of each render pass.
    VkViewport pending_viewport;
    VkRect2D   pending_scissor;
    bool       has_viewport;

    // Device name
    std::string device_name;

    // Cached physical-device limits exposed to Python.
    uint32_t max_3d_texture_size;

    // Ray-tracing capabilities. raytracing_supported is true only when the
    // VK_KHR_acceleration_structure + VK_KHR_ray_query extensions were
    // available *and* the matching features were enabled at device creation.
    bool raytracing_supported;

    // VK_EXT_metal_objects interop (macOS / MoltenVK only). When true, the
    // underlying MTLDevice and MTLCommandQueue can be exported for use by
    // the _metal_rt native ray tracing module.
    bool metal_objects_supported;

    // Timeline semaphore for Vulkan ↔ Metal RT GPU-GPU synchronization.
    // flush_gpu signals with rt_sync_value; the Metal RT dispatch waits on
    // that value and signals rt_sync_value+1; end_frame waits on rt_sync_value+1.
    VkSemaphore rt_timeline_semaphore;
    uint64_t    rt_sync_value;        // monotonically increasing
    bool        rt_sync_pending;      // true when Metal RT has signalled and
                                      // the next Vulkan submit must wait

    // Function pointers for KHR_acceleration_structure / KHR_ray_query.
    // Resolved once at device creation when raytracing_supported is true.
    PFN_vkGetAccelerationStructureBuildSizesKHR    pfn_get_as_build_sizes;
    PFN_vkCreateAccelerationStructureKHR           pfn_create_as;
    PFN_vkDestroyAccelerationStructureKHR          pfn_destroy_as;
    PFN_vkCmdBuildAccelerationStructuresKHR        pfn_cmd_build_as;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_get_as_device_address;

    // Debug messenger (validation layers in debug builds)
#ifndef NDEBUG
    VkDebugUtilsMessengerEXT debug_messenger;
#endif
};

// ---------------------------------------------------------------------------
//  Helper to extract context pointer from Python keyword args.
// ---------------------------------------------------------------------------
static inline VulkanContext* get_vulkan_context(PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    static const char* kwlist[] = {"context", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &ctx_ptr))
        return nullptr;
    return reinterpret_cast<VulkanContext*>(ctx_ptr);
}

// Reset the deferred descriptor-binding state. Called when starting a new
// render pass so leftover textures/buffers from the previous pass don't
// leak into the next pass's descriptor set — relevant when a texture used
// for sampling in pass A is rendered into during pass B (otherwise the
// descriptor's recorded layout would conflict with the attachment usage).
inline void reset_descriptor_bindings(VulkanContext* ctx) {
    for (uint32_t i = 0; i < MAX_UBO_BINDINGS; ++i)
        ctx->bound_ubo_buffers[i] = nullptr;
    for (uint32_t i = 0; i < MAX_TEXTURE_BINDINGS; ++i)
        ctx->bound_textures[i] = nullptr;
    ctx->descriptor_dirty = true;
}

// Pick the per-frame deletion slot to push a soon-to-be-released
// resource onto. The rule: associate the destroy with the most recent
// submission that could reference the resource, so the next reuse of
// that slot's fence is what gates the actual vkDestroy.
//
// Inside a frame (between begin_frame and end_frame, while recording)
// the resource may be referenced by commands we just issued, which will
// be submitted with current_frame's fence at end_frame -- queue to
// current_frame.
//
// Outside a frame (between end_frame and the next begin_frame) the most
// recently submitted slot is the one we just left, current_frame - 1.
// That slot is still in flight; queue to it so the destroy waits for
// its fence to signal at the next reuse.
//
// Without this distinction, a destroy_texture called right after
// end_frame goes onto the wrong slot's queue and gets freed while a
// descriptor set from the in-flight frame still references it.
inline uint32_t deferred_delete_slot(VulkanContext* ctx) {
    if (ctx->in_frame) {
        return ctx->current_frame;
    }
    return (ctx->current_frame + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;
}

// Drain a per-frame pending-deletes queue. Caller must guarantee the
// corresponding frame's command buffer has finished executing on the GPU
// (e.g. via fence wait at begin_frame, or vkDeviceWaitIdle at teardown).
inline void flush_pending_deletes(VulkanContext* ctx, PendingDeletes& pd) {
    for (VkFramebuffer fb : pd.framebuffers)
        vkDestroyFramebuffer(ctx->device, fb, nullptr);
    pd.framebuffers.clear();

    for (VkRenderPass rp : pd.render_passes)
        vkDestroyRenderPass(ctx->device, rp, nullptr);
    pd.render_passes.clear();

    for (BufferWithAllocator* bwa : pd.buffers) {
        vmaDestroyBuffer(bwa->allocator, bwa->base.buffer, bwa->base.allocation);
        delete bwa;
    }
    pd.buffers.clear();

    for (TextureWithDevice* tex : pd.textures) {
        vkDestroySampler(tex->device, tex->sampler, nullptr);
        vkDestroyImageView(tex->device, tex->image_view, nullptr);
        vmaDestroyImage(tex->allocator, tex->image, tex->allocation);
        delete tex;
    }
    pd.textures.clear();
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
void create_dummy_resources(VulkanContext* ctx);
void destroy_dummy_resources(VulkanContext* ctx);
// Fill `info` with the dummy view+sampler appropriate for set-1 binding `i`.
void dummy_image_info_for_binding(VulkanContext* ctx, uint32_t binding,
                                  VkDescriptorImageInfo& info);

#endif // CHIMERAX_VULKAN_INTERNAL_H
