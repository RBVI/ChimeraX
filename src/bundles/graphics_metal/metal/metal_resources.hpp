// metal_resources.hpp
// Manages Metal resources such as buffers, textures, and pipeline states

#pragma once

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace chimerax {
namespace graphics {

// Forward declarations
class MetalContext;

/**
 * Structure containing information about a shader library
 */
struct ShaderLibraryInfo {
    id<MTLLibrary> library;
    std::string name;
    bool isDefault;
};

/**
 * Manages Metal resources for rendering
 */
class MetalResources {
public:
    MetalResources(MetalContext* context);
    ~MetalResources();
    
    // Initialization
    bool initialize();
    
    // Shader management
    id<MTLLibrary> defaultLibrary() const { return _defaultLibrary; }
    id<MTLLibrary> loadLibrary(const std::string& filename);
    id<MTLLibrary> loadLibraryFromSource(const std::string& source, const std::string& name);
    id<MTLFunction> loadFunction(const std::string& functionName, id<MTLLibrary> library = nil);
    
    // Pipeline state objects
    id<MTLRenderPipelineState> createRenderPipelineState(
        const std::string& vertexFunction,
        const std::string& fragmentFunction,
        MTLPixelFormat colorPixelFormat = MTLPixelFormatBGRA8Unorm,
        MTLPixelFormat depthPixelFormat = MTLPixelFormatDepth32Float,
        bool blendingEnabled = false);
    
    id<MTLDepthStencilState> createDepthStencilState(
        bool depthTestEnabled = true,
        bool depthWriteEnabled = true,
        MTLCompareFunction depthCompareFunction = MTLCompareFunctionLess);
    
    // Buffer management
    id<MTLBuffer> createBuffer(
        const void* data,
        size_t length,
        MTLResourceOptions options = MTLResourceStorageModeShared);
    
    // Texture management
    id<MTLTexture> createTexture(
        uint32_t width,
        uint32_t height,
        MTLPixelFormat pixelFormat = MTLPixelFormatRGBA8Unorm,
        MTLTextureUsage usage = MTLTextureUsageShaderRead,
        MTLStorageMode storageMode = MTLStorageModeShared);
        
    id<MTLTexture> createTextureFromImage(
        const std::string& filename,
        MTLTextureUsage usage = MTLTextureUsageShaderRead,
        bool generateMipmaps = true);

    id<MTLSamplerState> createSamplerState(
        MTLSamplerMinMagFilter minFilter = MTLSamplerMinMagFilterLinear,
        MTLSamplerMinMagFilter magFilter = MTLSamplerMinMagFilterLinear,
        MTLSamplerAddressMode addressMode = MTLSamplerAddressModeClampToEdge);
        
    // Resource cleanup - call when context is lost
    void releaseResources();

private:
    MetalContext* _context;
    
    // Default shader library
    id<MTLLibrary> _defaultLibrary;
    
    // Cache of loaded libraries
    std::vector<ShaderLibraryInfo> _libraries;
    
    // Cache of pipeline states
    std::unordered_map<std::string, id<MTLRenderPipelineState>> _renderPipelineStates;
    std::unordered_map<std::string, id<MTLDepthStencilState>> _depthStencilStates;
    std::unordered_map<std::string, id<MTLSamplerState>> _samplerStates;
    
    // Internal methods
    std::string generatePipelineStateKey(
        const std::string& vertexFunction,
        const std::string& fragmentFunction,
        MTLPixelFormat colorPixelFormat,
        MTLPixelFormat depthPixelFormat,
        bool blendingEnabled);
        
    std::string generateDepthStencilStateKey(
        bool depthTestEnabled,
        bool depthWriteEnabled,
        MTLCompareFunction depthCompareFunction);
        
    std::string generateSamplerStateKey(
        MTLSamplerMinMagFilter minFilter,
        MTLSamplerMinMagFilter magFilter,
        MTLSamplerAddressMode addressMode);
};

} // namespace graphics
} // namespace chimerax
