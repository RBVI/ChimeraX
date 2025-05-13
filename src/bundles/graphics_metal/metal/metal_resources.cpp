// metal_resources.cpp
// Implementation of Metal resource management for ChimeraX

#include "metal_resources.hpp"
#include "metal_context.hpp"
#include <iostream>
#include <sstream>

namespace chimerax {
namespace graphics {

MetalResources::MetalResources(MetalContext* context)
    : _context(context)
    , _defaultLibrary(nil)
{
}

MetalResources::~MetalResources()
{
    releaseResources();
}

bool MetalResources::initialize()
{
    id<MTLDevice> device = _context->device();
    if (!device) {
        return false;
    }
    
    // Load default library
    NSError* error = nil;
    _defaultLibrary = [device newDefaultLibrary];
    if (!_defaultLibrary) {
        std::cerr << "Failed to load default Metal library" << std::endl;
        return false;
    }
    
    // Add to libraries list
    ShaderLibraryInfo defaultLib;
    defaultLib.library = _defaultLibrary;
    defaultLib.name = "default";
    defaultLib.isDefault = true;
    _libraries.push_back(defaultLib);
    
    return true;
}

void MetalResources::releaseResources()
{
    // Release all render pipeline states
    for (auto& pair : _renderPipelineStates) {
        [pair.second release];
    }
    _renderPipelineStates.clear();
    
    // Release all depth stencil states
    for (auto& pair : _depthStencilStates) {
        [pair.second release];
    }
    _depthStencilStates.clear();
    
    // Release all sampler states
    for (auto& pair : _samplerStates) {
        [pair.second release];
    }
    _samplerStates.clear();
    
    // Release all shader libraries
    for (auto& lib : _libraries) {
        [lib.library release];
    }
    _libraries.clear();
    
    _defaultLibrary = nil;
}

id<MTLLibrary> MetalResources::loadLibrary(const std::string& filename)
{
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // Check if library already loaded
    for (const auto& lib : _libraries) {
        if (lib.name == filename) {
            return lib.library;
        }
    }
    
    // Load library from file
    NSString* path = [NSString stringWithUTF8String:filename.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];
    
    if (!library) {
        std::cerr << "Failed to load Metal library from file: " << filename << std::endl;
        if (error) {
            std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
        }
        return nil;
    }
    
    // Add to libraries list
    ShaderLibraryInfo libInfo;
    libInfo.library = library;
    libInfo.name = filename;
    libInfo.isDefault = false;
    _libraries.push_back(libInfo);
    
    return library;
}

id<MTLLibrary> MetalResources::loadLibraryFromSource(const std::string& source, const std::string& name)
{
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // Check if library already loaded
    for (const auto& lib : _libraries) {
        if (lib.name == name) {
            return lib.library;
        }
    }
    
    // Load library from source
    NSString* nsSource = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    [options setLanguageVersion:MTLLanguageVersion2_0];
    
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:nsSource options:options error:&error];
    
    [options release];
    
    if (!library) {
        std::cerr << "Failed to load Metal library from source: " << name << std::endl;
        if (error) {
            std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
        }
        return nil;
    }
    
    // Add to libraries list
    ShaderLibraryInfo libInfo;
    libInfo.library = library;
    libInfo.name = name;
    libInfo.isDefault = false;
    _libraries.push_back(libInfo);
    
    return library;
}

id<MTLFunction> MetalResources::loadFunction(const std::string& functionName, id<MTLLibrary> library)
{
    if (!library) {
        library = _defaultLibrary;
    }
    
    if (!library) {
        return nil;
    }
    
    NSString* nsFunctionName = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:nsFunctionName];
    
    if (!function) {
        std::cerr << "Failed to load Metal function: " << functionName << std::endl;
    }
    
    return function;
}

id<MTLRenderPipelineState> MetalResources::createRenderPipelineState(
    const std::string& vertexFunction,
    const std::string& fragmentFunction,
    MTLPixelFormat colorPixelFormat,
    MTLPixelFormat depthPixelFormat,
    bool blendingEnabled)
{
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // Generate key for caching
    std::string key = generatePipelineStateKey(
        vertexFunction, fragmentFunction, colorPixelFormat, depthPixelFormat, blendingEnabled);
    
    // Check if pipeline state already exists
    auto it = _renderPipelineStates.find(key);
    if (it != _renderPipelineStates.end()) {
        return it->second;
    }
    
    // Load shader functions
    id<MTLFunction> vertexFunc = loadFunction(vertexFunction);
    id<MTLFunction> fragmentFunc = loadFunction(fragmentFunction);
    
    if (!vertexFunc || !fragmentFunc) {
        return nil;
    }
    
    // Create pipeline descriptor
    MTLRenderPipelineDescriptor* pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.vertexFunction = vertexFunc;
    pipelineDescriptor.fragmentFunction = fragmentFunc;
    pipelineDescriptor.colorAttachments[0].pixelFormat = colorPixelFormat;
    pipelineDescriptor.depthAttachmentPixelFormat = depthPixelFormat;
    
    // Configure blending if enabled
    if (blendingEnabled) {
        pipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    }
    
    // Create pipeline state
    NSError* error = nil;
    id<MTLRenderPipelineState> pipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    
    [pipelineDescriptor release];
    [vertexFunc release];
    [fragmentFunc release];
    
    if (!pipelineState) {
        std::cerr << "Failed to create render pipeline state for vertex: " << vertexFunction
                  << ", fragment: " << fragmentFunction << std::endl;
        if (error) {
            std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
        }
        return nil;
    }
    
    // Cache pipeline state
    _renderPipelineStates[key] = pipelineState;
    
    return pipelineState;
}

id<MTLDepthStencilState> MetalResources::createDepthStencilState(
    bool depthTestEnabled,
    bool depthWriteEnabled,
    MTLCompareFunction depthCompareFunction)
{
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // Generate key for caching
    std::string key = generateDepthStencilStateKey(
        depthTestEnabled, depthWriteEnabled, depthCompareFunction);
    
    // Check if depth stencil state already exists
    auto it = _depthStencilStates.find(key);
    if (it != _depthStencilStates.end()) {
        return it->second;
    }
    
    // Create depth stencil descriptor
    MTLDepthStencilDescriptor* depthStencilDescriptor = [[MTLDepthStencilDescriptor alloc] init];
    depthStencilDescriptor.depthCompareFunction = depthTestEnabled ? depthCompareFunction : MTLCompareFunctionAlways;
    depthStencilDescriptor.depthWriteEnabled = depthWriteEnabled;
    
    // Create depth stencil state
    id<MTLDepthStencilState> depthStencilState = [device newDepthStencilStateWithDescriptor:depthStencilDescriptor];
    
    [depthStencilDescriptor release];
    
    if (!depthStencilState) {
        std::cerr << "Failed to create depth stencil state" << std::endl;
        return nil;
    }
    
    // Cache depth stencil state
    _depthStencilStates[key] = depthStencilState;
    
    return depthStencilState;
}

id<MTLBuffer> MetalResources::createBuffer(
    const void* data,
    size_t length,
    MTLResourceOptions options)
{
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    id<MTLBuffer> buffer = nil;
    
    if (data) {
        buffer = [device newBufferWithBytes:data length:length options:options];
    } else {
        buffer = [device newBufferWithLength:length options:options];
    }
    
    if (!buffer) {
        std::cerr << "Failed to create Metal buffer of size: " << length << std::endl;
    }
    
    return buffer;
}

id<MTLTexture> MetalResources::createTexture(
    uint32_t width,
    uint32_t height,
    MTLPixelFormat pixelFormat,
    MTLTextureUsage usage,
    MTLStorageMode storageMode)
{
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    MTLTextureDescriptor* textureDescriptor = [[MTLTextureDescriptor alloc] init];
    textureDescriptor.textureType = MTLTextureType2D;
    textureDescriptor.pixelFormat = pixelFormat;
    textureDescriptor.width = width;
    textureDescriptor.height = height;
    textureDescriptor.usage = usage;
    textureDescriptor.storageMode = storageMode;
    
    id<MTLTexture> texture = [device newTextureWithDescriptor:textureDescriptor];
    
    [textureDescriptor release];
    
    if (!texture) {
        std::cerr << "Failed to create Metal texture of size: " << width << "x" << height << std::endl;
    }
    
    return texture;
}

id<MTLTexture> MetalResources::createTextureFromImage(
    const std::string& filename,
    MTLTextureUsage usage,
    bool generateMipmaps)
{
    // In a real implementation, this would load an image from disk
    // and create a texture from it. For simplicity, we'll just create
    // a placeholder implementation that returns nil.
    std::cerr << "createTextureFromImage not implemented" << std::endl;
    return nil;
}

id<MTLSamplerState> MetalResources::createSamplerState(
    MTLSamplerMinMagFilter minFilter,
    MTLSamplerMinMagFilter magFilter,
    MTLSamplerAddressMode addressMode)
{
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    // Generate key for caching
    std::string key = generateSamplerStateKey(minFilter, magFilter, addressMode);
    
    // Check if sampler state already exists
    auto it = _samplerStates.find(key);
    if (it != _samplerStates.end()) {
        return it->second;
    }
    
    // Create sampler descriptor
    MTLSamplerDescriptor* samplerDescriptor = [[MTLSamplerDescriptor alloc] init];
    samplerDescriptor.minFilter = minFilter;
    samplerDescriptor.magFilter = magFilter;
    samplerDescriptor.sAddressMode = addressMode;
    samplerDescriptor.tAddressMode = addressMode;
    
    // Create sampler state
    id<MTLSamplerState> samplerState = [device newSamplerStateWithDescriptor:samplerDescriptor];
    
    [samplerDescriptor release];
    
    if (!samplerState) {
        std::cerr << "Failed to create Metal sampler state" << std::endl;
        return nil;
    }
    
    // Cache sampler state
    _samplerStates[key] = samplerState;
    
    return samplerState;
}

std::string MetalResources::generatePipelineStateKey(
    const std::string& vertexFunction,
    const std::string& fragmentFunction,
    MTLPixelFormat colorPixelFormat,
    MTLPixelFormat depthPixelFormat,
    bool blendingEnabled)
{
    std::stringstream ss;
    ss << vertexFunction << "_" << fragmentFunction << "_"
       << colorPixelFormat << "_" << depthPixelFormat << "_"
       << (blendingEnabled ? "blend" : "noblend");
    return ss.str();
}

std::string MetalResources::generateDepthStencilStateKey(
    bool depthTestEnabled,
    bool depthWriteEnabled,
    MTLCompareFunction depthCompareFunction)
{
    std::stringstream ss;
    ss << (depthTestEnabled ? "test" : "notest") << "_"
       << (depthWriteEnabled ? "write" : "nowrite") << "_"
       << depthCompareFunction;
    return ss.str();
}

std::string MetalResources::generateSamplerStateKey(
    MTLSamplerMinMagFilter minFilter,
    MTLSamplerMinMagFilter magFilter,
    MTLSamplerAddressMode addressMode)
{
    std::stringstream ss;
    ss << minFilter << "_" << magFilter << "_" << addressMode;
    return ss.str();
}

} // namespace graphics
} // namespace chimerax
