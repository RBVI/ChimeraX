// metal_renderer.hpp
// Core Metal rendering functionality for ChimeraX

#pragma once

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <memory>
#include <vector>

namespace chimerax {
namespace graphics {

// Forward declarations
class MetalContext;
class MetalScene;

/**
 * Common uniforms for all shaders - should match shaders in metal_shaders.metal
 */
struct Uniforms {
    // Matrices
    simd::float4x4 modelMatrix;
    simd::float4x4 viewMatrix;
    simd::float4x4 projectionMatrix;
    simd::float4x4 normalMatrix;
    
    // Camera
    simd::float3 cameraPosition;
    float padding1;
    
    // Lighting
    simd::float3 lightPosition;
    float lightRadius;
    simd::float3 lightColor;
    float lightIntensity;
    simd::float3 ambientColor;
    float ambientIntensity;
};

/**
 * Material properties for various rendering modes
 */
struct MaterialProperties {
    // Basic properties
    simd::float4 color;
    float roughness;
    float metallic;
    float ambientOcclusion;
    float padding;
    
    // Additional properties for molecular rendering
    float atomRadius;
    float bondRadius;
    float outlineWidth;
    float outlineStrength;
};

/**
 * Core renderer for Metal-based graphics in ChimeraX
 */
class MetalRenderer {
public:
    // Constructor
    MetalRenderer(MetalContext* context);
    ~MetalRenderer();
    
    // Initialization
    bool initialize();
    
    // Rendering methods
    void beginFrame();
    void endFrame();
    
    // Scene management
    void setScene(MetalScene* scene) { _scene = scene; }
    MetalScene* scene() const { return _scene; }
    
    // Rendering of basic elements
    void renderSpheres(
        id<MTLBuffer> positionBuffer,
        id<MTLBuffer> colorBuffer,
        id<MTLBuffer> radiusBuffer,
        uint32_t count);
        
    void renderCylinders(
        id<MTLBuffer> startPositionBuffer,
        id<MTLBuffer> endPositionBuffer,
        id<MTLBuffer> colorBuffer,
        id<MTLBuffer> radiusBuffer,
        uint32_t count);
        
    void renderTriangles(
        id<MTLBuffer> vertexBuffer,
        id<MTLBuffer> colorBuffer,
        id<MTLBuffer> normalBuffer,
        id<MTLBuffer> indexBuffer,
        uint32_t indexCount);

private:
    MetalContext* _context;
    MetalScene* _scene;
    
    // Metal pipeline objects
    id<MTLRenderPipelineState> _spherePipelineState;
    id<MTLRenderPipelineState> _cylinderPipelineState;
    id<MTLRenderPipelineState> _trianglePipelineState;
    id<MTLDepthStencilState> _defaultDepthState;
    
    // Uniform buffers
    id<MTLBuffer> _uniformBuffer;
    
    // Internal methods
    bool createPipelines();
    void updateUniforms(id<MTLCommandBuffer> commandBuffer);
};

} // namespace graphics
} // namespace chimerax
