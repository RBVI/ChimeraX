/*
 * MoltenVK <-> Metal RT interop via VK_EXT_metal_objects.
 *
 * Exports the MTLDevice and MTLCommandQueue from the Vulkan renderer's
 * MoltenVK device so we can use them for native Metal ray tracing.
 */

#include <Python.h>
#include "metal_rt_internal.h"
#include "vulkan_internal.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#include <vulkan/vulkan_metal.h>

// Global RT context — one per process (matches the single-window model).
static MetalRTContext* s_rt_ctx = nullptr;

// Stashed references for the surface layer so we can resize it.
static NSView* s_surface_view = nil;
static CAMetalLayer* s_surface_layer = nil;

// ---------------------------------------------------------------------------
//  Create a VkSurfaceKHR from an NSView* via CAMetalLayer.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_rt_create_surface(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t instance_handle;
    Py_ssize_t window_handle;
    static const char* kwlist[] = {"instance", "window_handle", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist,
            &instance_handle, &window_handle))
        return NULL;

    VkInstance instance = reinterpret_cast<VkInstance>(instance_handle);
    NSView* view = reinterpret_cast<NSView*>(window_handle);

    [view setWantsLayer:YES];
    CAMetalLayer* metal_layer = [CAMetalLayer layer];
    metal_layer.contentsScale = view.window.backingScaleFactor;
    metal_layer.frame = view.bounds;
    [view setLayer:metal_layer];

    // Stash for resize_surface.
    s_surface_view = view;
    s_surface_layer = metal_layer;

    VkMetalSurfaceCreateInfoEXT surf_info{};
    surf_info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    surf_info.pLayer = metal_layer;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult vr = vkCreateMetalSurfaceEXT(instance, &surf_info, nullptr, &surface);
    if (vr != VK_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "vkCreateMetalSurfaceEXT failed");
        return NULL;
    }

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(surface));
}

// ---------------------------------------------------------------------------
//  Update the CAMetalLayer's bounds to match the current view size.
//  Must be called before swapchain recreation so MoltenVK sees the
//  correct drawableSize in VkSurfaceCapabilitiesKHR.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_rt_resize_surface(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t window_handle = 0;
    static const char* kwlist[] = {"window_handle", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("|n"), (char**)kwlist, &window_handle))
        return NULL;

    NSView* view = window_handle
        ? reinterpret_cast<NSView*>(window_handle) : s_surface_view;
    CAMetalLayer* layer = nil;
    if (view && [view.layer isKindOfClass:[CAMetalLayer class]])
        layer = (CAMetalLayer*)view.layer;
    if (!layer && view == s_surface_view)
        layer = s_surface_layer;

    if (view && layer) {
        CGFloat scale = view.window.backingScaleFactor;
        NSRect bounds = view.bounds;
        layer.frame = bounds;
        layer.contentsScale = scale;
        // Explicitly set drawableSize so MoltenVK reports the correct
        // currentExtent in VkSurfaceCapabilitiesKHR immediately.
        layer.drawableSize = CGSizeMake(
            bounds.size.width * scale, bounds.size.height * scale);
    }
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Initialize from VulkanContext — export Metal objects via MoltenVK.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_rt_init(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    static const char* kwlist[] = {"context", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &ctx_ptr))
        return NULL;

    VulkanContext* vk_ctx = reinterpret_cast<VulkanContext*>(ctx_ptr);

    if (!vk_ctx->metal_objects_supported) {
        PyErr_SetString(PyExc_RuntimeError,
                        "VK_EXT_metal_objects not available on this device");
        return NULL;
    }

    // Resolve the export function pointer.
    auto pfn = (PFN_vkExportMetalObjectsEXT)vkGetDeviceProcAddr(
        vk_ctx->device, "vkExportMetalObjectsEXT");
    if (!pfn) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to resolve vkExportMetalObjectsEXT");
        return NULL;
    }

    // Export MTLDevice.
    VkExportMetalDeviceInfoEXT deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT;
    deviceInfo.mtlDevice = nil;

    VkExportMetalObjectsInfoEXT exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
    exportInfo.pNext = &deviceInfo;

    pfn(vk_ctx->device, &exportInfo);

    if (!deviceInfo.mtlDevice) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to export MTLDevice from MoltenVK");
        return NULL;
    }

    // Export MTLCommandQueue from the graphics queue.
    VkExportMetalCommandQueueInfoEXT queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT;
    queueInfo.queue = vk_ctx->graphics_queue;
    queueInfo.mtlCommandQueue = nil;

    exportInfo.pNext = &queueInfo;
    pfn(vk_ctx->device, &exportInfo);

    if (!queueInfo.mtlCommandQueue) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to export MTLCommandQueue from MoltenVK");
        return NULL;
    }

    // Check that the device supports ray tracing.
    if (![deviceInfo.mtlDevice supportsRaytracing]) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Metal device does not support ray tracing");
        return NULL;
    }

    // Export the timeline semaphore as MTLSharedEvent for GPU-GPU sync.
    id<MTLSharedEvent> syncEvent = nil;
    if (vk_ctx->rt_timeline_semaphore != VK_NULL_HANDLE) {
        VkExportMetalSharedEventInfoEXT eventInfo{};
        eventInfo.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT;
        eventInfo.semaphore = vk_ctx->rt_timeline_semaphore;
        eventInfo.mtlSharedEvent = nil;

        VkExportMetalObjectsInfoEXT eventExport{};
        eventExport.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
        eventExport.pNext = &eventInfo;
        pfn(vk_ctx->device, &eventExport);
        syncEvent = eventInfo.mtlSharedEvent;
        fprintf(stderr, "[MetalRT] Exported MTLSharedEvent: %p\n", syncEvent);
    } else {
        fprintf(stderr, "[MetalRT] No timeline semaphore to export\n");
    }

    // Create or refresh the RT context. On re-init (a new Vulkan context
    // after the old one was destroyed) the device and queue MUST be
    // updated too: the previously exported MTLCommandQueue belongs to the
    // destroyed VkDevice, and any Metal call through the stale object is
    // a use-after-free.
    if (!s_rt_ctx)
        s_rt_ctx = new MetalRTContext();
    s_rt_ctx->device = deviceInfo.mtlDevice;
    s_rt_ctx->commandQueue = queueInfo.mtlCommandQueue;
    s_rt_ctx->vk_ctx = vk_ctx;
    s_rt_ctx->pfn_export_metal_objects = pfn;
    s_rt_ctx->syncEvent = syncEvent;

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(s_rt_ctx));
}

// ---------------------------------------------------------------------------
//  Export a VulkanTexture as a MetalRTTexture via VK_EXT_metal_objects.
//
//  Takes a VulkanTexture* handle (from _vulkan) and returns a
//  MetalRTTexture* handle that _metal_rt dispatch functions can use.
//  The caller must free the MetalRTTexture with metal_rt_release_texture.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_rt_export_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr, tex_ptr;
    static const char* kwlist[] = {"context", "texture", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nn"), (char**)kwlist,
            &ctx_ptr, &tex_ptr))
        return NULL;

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    VulkanTexture* vk_tex = reinterpret_cast<VulkanTexture*>(tex_ptr);

    // Export the VkImage as MTLTexture.
    VkExportMetalTextureInfoEXT texInfo{};
    texInfo.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT;
    texInfo.image = vk_tex->image;
    texInfo.imageView = VK_NULL_HANDLE;
    texInfo.bufferView = VK_NULL_HANDLE;
    texInfo.plane = VK_IMAGE_ASPECT_COLOR_BIT;
    texInfo.mtlTexture = nil;

    VkExportMetalObjectsInfoEXT exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
    exportInfo.pNext = &texInfo;

    ctx->pfn_export_metal_objects(ctx->vk_ctx->device, &exportInfo);

    if (!texInfo.mtlTexture) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to export VkImage as MTLTexture");
        return NULL;
    }

    MetalRTTexture* rt_tex = new MetalRTTexture();
    rt_tex->texture = texInfo.mtlTexture;
    rt_tex->sampler = nil;
    rt_tex->pixelFormat = texInfo.mtlTexture.pixelFormat;
    rt_tex->width = vk_tex->width;
    rt_tex->height = vk_tex->height;

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(rt_tex));
}

extern "C"
PyObject* metal_rt_release_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t tex_ptr;
    static const char* kwlist[] = {"texture", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("n"), (char**)kwlist, &tex_ptr))
        return NULL;

    if (tex_ptr) {
        MetalRTTexture* rt_tex = reinterpret_cast<MetalRTTexture*>(tex_ptr);
        delete rt_tex;
    }
    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Export a VkDeviceMemory as an MTLBuffer via VK_EXT_metal_objects.
//
//  Takes the VkDeviceMemory handle directly (obtained by _vulkan from VMA).
//  Returns the MTLBuffer pointer as a long for use in dispatch calls.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_rt_export_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    Py_ssize_t ctx_ptr;
    unsigned long long memory_handle;
    static const char* kwlist[] = {"context", "device_memory", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("nK"), (char**)kwlist,
            &ctx_ptr, &memory_handle))
        return NULL;

    MetalRTContext* ctx = reinterpret_cast<MetalRTContext*>(ctx_ptr);
    VkDeviceMemory memory = reinterpret_cast<VkDeviceMemory>(memory_handle);

    VkExportMetalBufferInfoEXT bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT;
    bufInfo.memory = memory;
    bufInfo.mtlBuffer = nil;

    VkExportMetalObjectsInfoEXT exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
    exportInfo.pNext = &bufInfo;

    ctx->pfn_export_metal_objects(ctx->vk_ctx->device, &exportInfo);

    if (!bufInfo.mtlBuffer) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to export VkDeviceMemory as MTLBuffer");
        return NULL;
    }

    return PyLong_FromSsize_t(reinterpret_cast<Py_ssize_t>(
        (__bridge void*)bufInfo.mtlBuffer));
}

// ---------------------------------------------------------------------------
//  Tear down the RT context.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_rt_destroy(PyObject* self, PyObject* args, PyObject* kwds) {
    if (s_rt_ctx) {
        delete s_rt_ctx;
        s_rt_ctx = nullptr;
    }
    Py_RETURN_NONE;
}
