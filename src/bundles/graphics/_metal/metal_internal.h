/*
 * Internal header shared between Metal .mm files.
 * Defines the MetalContext struct so all files can access its fields.
 */

#ifndef CHIMERAX_METAL_INTERNAL_H
#define CHIMERAX_METAL_INTERNAL_H

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <AppKit/AppKit.h>

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    CAMetalLayer *layer;
    id<MTLLibrary> shaderLibrary;

    // Per-frame state
    id<CAMetalDrawable> currentDrawable;
    id<MTLCommandBuffer> currentCommandBuffer;
    id<MTLRenderCommandEncoder> currentEncoder;

    // Depth buffer
    id<MTLTexture> depthTexture;
    MTLPixelFormat depthPixelFormat;

    // Size
    NSUInteger width;
    NSUInteger height;
};

struct MetalBuffer {
    id<MTLBuffer> buffer;
    NSUInteger size;
};

struct MetalPipeline {
    id<MTLRenderPipelineState> state;
    id<MTLDepthStencilState> depthStencilState;
};

struct MetalTexture {
    id<MTLTexture> texture;
    id<MTLSamplerState> sampler;
    MTLPixelFormat pixelFormat;
    NSUInteger width;
    NSUInteger height;
};

// Helper to extract context from Python args
static inline MetalContext* get_metal_context(long ptr) {
    return (MetalContext*)ptr;
}

#endif // CHIMERAX_METAL_INTERNAL_H
