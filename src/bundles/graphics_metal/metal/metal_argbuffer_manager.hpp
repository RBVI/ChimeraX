// metal_argbuffer_manager.hpp
// Manages argument buffers for efficient resource binding in Metal

#pragma once

#import <Metal/Metal.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace chimerax {
namespace graphics_metal {

// Forward declarations
class MetalContext;

/**
 * Type of argument buffer encoder
 */
enum class ArgBufferType {
    Vertex,
    Fragment,
    Compute,
    RayData
};

/**
 * Represents a single argument buffer layout and associated encoder
 */
class MetalArgBuffer {
public:
    MetalArgBuffer(id<MTLDevice> device, id<MTLArgumentEncoder> encoder, const std::string& name);
    ~MetalArgBuffer();
    
    // Buffer access
    id<MTLBuffer> buffer() const { return _buffer; }
    id<MTLArgumentEncoder> encoder() const { return _encoder; }
    
    // Set resources into the argument buffer
    void setBuffer(uint32_t index, id<MTLBuffer> buffer, uint32_t offset = 0);
    void setTexture(uint32_t index, id<MTLTexture> texture);
    void setSamplerState(uint32_t index, id<MTLSamplerState> sampler);
    void setAccelerationStructure(uint32_t index, id<MTLAccelerationStructure> accelStructure);
    
    // Get the size of the argument buffer
    NSUInteger encodedLength() const;
    
    // Get the name of the argument buffer
    const std::string& name() const { return _name; }
    
private:
    id<MTLDevice> _device;
    id<MTLArgumentEncoder> _encoder;
    id<MTLBuffer> _buffer;
    std::string _name;
};

/**
 * Manages argument buffers for efficient resource binding
 */
class MetalArgBufferManager {
public:
    MetalArgBufferManager(MetalContext* context);
    ~MetalArgBufferManager();
    
    // Initialization
    bool initialize();
    
    // Create argument buffer from shader function
    std::shared_ptr<MetalArgBuffer> createArgBuffer(
        const std::string& functionName, 
        ArgBufferType type,
        id<MTLLibrary> library = nil);
    
    // Create argument buffer from buffer structure (manual encoding)
    std::shared_ptr<MetalArgBuffer> createArgBufferFromLayout(
        const std::vector<MTLArgumentDescriptor*>& descriptors,
        const std::string& name);
    
    // Get existing argument buffer by name
    std::shared_ptr<MetalArgBuffer> getArgBuffer(const std::string& name) const;
    
    // Clear all argument buffers (useful when context is lost)
    void clearArgBuffers();
    
private:
    MetalContext* _context;
    
    // Cache of argument buffers
    std::unordered_map<std::string, std::shared_ptr<MetalArgBuffer>> _argBuffers;
};

} // namespace graphics_metal
} // namespace chimerax
