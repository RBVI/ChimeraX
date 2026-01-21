// metal_renderer.cpp
// Implementation of Metal rendering for ChimeraX

#include "metal_renderer.hpp"
#include "metal_context.hpp"
#include "metal_resources.hpp"
#include "metal_scene.hpp"
#include <iostream>

namespace chimerax {
namespace graphics {

MetalRenderer::MetalRenderer(MetalContext* context)
    : _context(context)
    , _scene(nullptr)
    , _spherePipelineState(nil)
    , _cylinderPipelineState(nil)
    , _trianglePipelineState(nil)
    , _defaultDepthState(nil)
    , _uniformBuffer(nil)
{
}

MetalRenderer::~MetalRenderer()
{
    if (_uniformBuffer) {
        [_uniformBuffer release];
    }
    
    // Pipeline states are owned by the resources manager
}

bool MetalRenderer::initialize()
{
    // Check context
    if (!_context || !_context->isInitialized()) {
        std::cerr << "MetalRenderer::initialize: No valid Metal context" << std::endl;
        return false;
    }

    // Create uniform buffer
    _uniformBuffer = _context->resources()->createBuffer(
        nullptr, sizeof(Uniforms), MTLResourceStorageModeShared);
        
    if (!_uniformBuffer) {
        std::cerr << "MetalRenderer::initialize: Failed to create uniform buffer" << std::endl;
        return false;
    }

    // Create render pipelines
    if (!createPipelines()) {
        std::cerr << "MetalRenderer::initialize: Failed to create render pipelines" << std::endl;
        return false;
    }

    return true;
}

bool MetalRenderer::createPipelines()
{
    MetalResources* resources = _context->resources();
    
    // Create default depth state
    _defaultDepthState = resources->createDepthStencilState(
        true, true, MTLCompareFunctionLess);
    
    if (!_defaultDepthState) {
        return false;
    }
    
    // Create sphere pipeline
    _spherePipelineState = resources->createRenderPipelineState(
        "vertexSphere", "fragmentSphere",
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, true);
    
    if (!_spherePipelineState) {
        return false;
    }
    
    // Create cylinder pipeline
    _cylinderPipelineState = resources->createRenderPipelineState(
        "vertexCylinder", "fragmentCylinder",
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, true);
    
    if (!_cylinderPipelineState) {
        return false;
    }
    
    // Create triangle pipeline
    _trianglePipelineState = resources->createRenderPipelineState(
        "vertexTriangle", "fragmentTriangle",
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, true);
    
    if (!_trianglePipelineState) {
        return false;
    }
    
    return true;
}

void MetalRenderer::beginFrame()
{
    // Clear the uniform buffer data
    Uniforms* uniforms = static_cast<Uniforms*>([_uniformBuffer contents]);
    if (uniforms) {
        // Initialize with default values
        uniforms->modelMatrix = simd::float4x4(1.0f);
        uniforms->viewMatrix = simd::float4x4(1.0f);
        uniforms->projectionMatrix = simd::float4x4(1.0f);
        uniforms->normalMatrix = simd::float4x4(1.0f);
        
        uniforms->cameraPosition = simd::float3(0.0f, 0.0f, 5.0f);
        
        uniforms->lightPosition = simd::float3(0.0f, 5.0f, 5.0f);
        uniforms->lightRadius = 50.0f;
        uniforms->lightColor = simd::float3(1.0f, 1.0f, 1.0f);
        uniforms->lightIntensity = 1.0f;
        uniforms->ambientColor = simd::float3(0.1f, 0.1f, 0.1f);
        uniforms->ambientIntensity = 1.0f;
        
        // If we have a scene, update with scene values
        if (_scene) {
            // TODO: Get actual values from scene
        }
    }
}

void MetalRenderer::endFrame()
{
    // Nothing to do here yet
}

void MetalRenderer::updateUniforms(id<MTLCommandBuffer> commandBuffer)
{
    // In a real implementation, this would update uniform values
    // from the current scene/camera state. For now, this is just a placeholder.
}

void MetalRenderer::renderSpheres(
    id<MTLBuffer> positionBuffer,
    id<MTLBuffer> colorBuffer,
    id<MTLBuffer> radiusBuffer,
    uint32_t count)
{
    if (!_context || !positionBuffer || count == 0) {
        return;
    }
    
    MTKView* view = _context->mtkView();
    if (!view) {
        return;
    }
    
    id<MTLCommandQueue> commandQueue = _context->commandQueue();
    id<CAMetalDrawable> drawable = [view currentDrawable];
    MTLRenderPassDescriptor* renderPassDescriptor = [view currentRenderPassDescriptor];
    
    if (!commandQueue || !drawable || !renderPassDescriptor) {
        return;
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    [commandBuffer setLabel:@"Sphere Render Command Buffer"];
    
    // Update uniforms
    updateUniforms(commandBuffer);
    
    // Create render command encoder
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder setLabel:@"Sphere Render Encoder"];
    
    // Set render pipeline
    [renderEncoder setRenderPipelineState:_spherePipelineState];
    [renderEncoder setDepthStencilState:_defaultDepthState];
    
    // Set buffers
    [renderEncoder setVertexBuffer:positionBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:colorBuffer offset:0 atIndex:1];
    [renderEncoder setVertexBuffer:radiusBuffer offset:0 atIndex:2];
    [renderEncoder setVertexBuffer:_uniformBuffer offset:0 atIndex:3];
    [renderEncoder setFragmentBuffer:_uniformBuffer offset:0 atIndex:0];
    
    // Draw spheres
    [renderEncoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:count];
    
    // End encoding and submit
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

void MetalRenderer::renderCylinders(
    id<MTLBuffer> startPositionBuffer,
    id<MTLBuffer> endPositionBuffer,
    id<MTLBuffer> colorBuffer,
    id<MTLBuffer> radiusBuffer,
    uint32_t count)
{
    if (!_context || !startPositionBuffer || !endPositionBuffer || count == 0) {
        return;
    }
    
    MTKView* view = _context->mtkView();
    if (!view) {
        return;
    }
    
    id<MTLCommandQueue> commandQueue = _context->commandQueue();
    id<CAMetalDrawable> drawable = [view currentDrawable];
    MTLRenderPassDescriptor* renderPassDescriptor = [view currentRenderPassDescriptor];
    
    if (!commandQueue || !drawable || !renderPassDescriptor) {
        return;
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    [commandBuffer setLabel:@"Cylinder Render Command Buffer"];
    
    // Update uniforms
    updateUniforms(commandBuffer);
    
    // Create render command encoder
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder setLabel:@"Cylinder Render Encoder"];
    
    // Set render pipeline
    [renderEncoder setRenderPipelineState:_cylinderPipelineState];
    [renderEncoder setDepthStencilState:_defaultDepthState];
    
    // Set buffers
    [renderEncoder setVertexBuffer:startPositionBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:endPositionBuffer offset:0 atIndex:1];
    [renderEncoder setVertexBuffer:colorBuffer offset:0 atIndex:2];
    [renderEncoder setVertexBuffer:radiusBuffer offset:0 atIndex:3];
    [renderEncoder setVertexBuffer:_uniformBuffer offset:0 atIndex:4];
    [renderEncoder setFragmentBuffer:_uniformBuffer offset:0 atIndex:0];
    
    // Draw cylinders (as lines with geometry shader)
    [renderEncoder drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:count * 2];
    
    // End encoding and submit
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

void MetalRenderer::renderTriangles(
    id<MTLBuffer> vertexBuffer,
    id<MTLBuffer> colorBuffer,
    id<MTLBuffer> normalBuffer,
    id<MTLBuffer> indexBuffer,
    uint32_t indexCount)
{
    if (!_context || !vertexBuffer || !indexBuffer || indexCount == 0) {
        return;
    }
    
    MTKView* view = _context->mtkView();
    if (!view) {
        return;
    }
    
    id<MTLCommandQueue> commandQueue = _context->commandQueue();
    id<CAMetalDrawable> drawable = [view currentDrawable];
    MTLRenderPassDescriptor* renderPassDescriptor = [view currentRenderPassDescriptor];
    
    if (!commandQueue || !drawable || !renderPassDescriptor) {
        return;
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    [commandBuffer setLabel:@"Triangle Render Command Buffer"];
    
    // Update uniforms
    updateUniforms(commandBuffer);
    
    // Create render command encoder
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder setLabel:@"Triangle Render Encoder"];
    
    // Set render pipeline
    [renderEncoder setRenderPipelineState:_trianglePipelineState];
    [renderEncoder setDepthStencilState:_defaultDepthState];
    
    // Set buffers
    [renderEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:colorBuffer offset:0 atIndex:1];
    [renderEncoder setVertexBuffer:normalBuffer offset:0 atIndex:2];
    [renderEncoder setVertexBuffer:_uniformBuffer offset:0 atIndex:3];
    [renderEncoder setFragmentBuffer:_uniformBuffer offset:0 atIndex:0];
    
    // Draw triangles
    [renderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                              indexCount:indexCount
                               indexType:MTLIndexTypeUInt32
                             indexBuffer:indexBuffer
                       indexBufferOffset:0];
    
    // End encoding and submit
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

} // namespace graphics
} // namespace chimerax
