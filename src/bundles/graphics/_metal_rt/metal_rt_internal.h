/*
 * Internal header for the Metal RT interop module.
 *
 * This module provides native Metal ray tracing (acceleration structures,
 * RTAO, path tracer) on macOS when the Vulkan renderer runs through
 * MoltenVK. The MTLDevice and MTLCommandQueue are exported from MoltenVK
 * via VK_EXT_metal_objects -- no separate Metal context is created.
 */

#ifndef CHIMERAX_METAL_RT_INTERNAL_H
#define CHIMERAX_METAL_RT_INTERNAL_H

#import <Metal/Metal.h>

// Forward-declare the Vulkan types we need without pulling in the full
// Vulkan header (which would conflict with MoltenVK's ObjC environment
// in some configurations).
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

struct VulkanContext;

// Minimal context for Metal ray tracing via MoltenVK interop.
struct MetalRTContext {
    id<MTLDevice>       device;         // Exported from MoltenVK
    id<MTLCommandQueue> commandQueue;   // Exported from MoltenVK

    // Back-pointer to the owning VulkanContext for interop calls.
    VulkanContext*       vk_ctx;

    // VK_EXT_metal_objects function pointer (resolved once at init).
    PFN_vkExportMetalObjectsEXT pfn_export_metal_objects;

    // Shared event for GPU-GPU sync with Vulkan timeline semaphore.
    id<MTLSharedEvent> syncEvent;
};

// Texture wrapper matching the Metal backend's MetalTexture struct layout.
// Used by RTAO/pathtracer dispatch when receiving texture handles from Python.
struct MetalRTTexture {
    id<MTLTexture>      texture;
    id<MTLSamplerState> sampler;
    MTLPixelFormat      pixelFormat;
    NSUInteger          width;
    NSUInteger          height;
};

#endif // CHIMERAX_METAL_RT_INTERNAL_H
