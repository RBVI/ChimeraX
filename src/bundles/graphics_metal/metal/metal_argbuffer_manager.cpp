// metal_argbuffer_manager.cpp
// Implementation of argument buffers for efficient resource binding in Metal

#include "metal_argbuffer_manager.hpp"
#include "metal_context.hpp"
#include "metal_resources.hpp"
#include <iostream>

namespace chimerax {
namespace graphics_metal {

// MetalArgBuffer implementation
MetalArgBuffer::MetalArgBuffer(id<MTLDevice> device, id<MTLArgumentEncoder> encoder, const std::string& name)
    : _device(device)
    , _encoder(encoder)
    , _buffer(nil)
    , _name(name)
{
    if (_encoder) {
        // Retain encoder
        [_encoder retain];
        
        // Create buffer to hold argument data
        NSUInteger length = [_encoder encodedLength];
        _buffer = [_device newBufferWithLength:length options:MTLResourceStorageModeShared];
        
        if (_buffer) {
            [_buffer setLabel:[NSString stringWithFormat:@"ArgBuffer: %s", name.c_str()]];
            
            // Initialize the buffer with the encoder
            [_encoder setArgumentBuffer:_buffer offset:0];
        }
    }
}

MetalArgBuffer::~MetalArgBuffer()
{
    if (_buffer) {
        [_buffer release];
        _buffer = nil;
    }
    
    if (_encoder) {
        [_encoder release];
        _encoder = nil;
    }
}

void MetalArgBuffer::setBuffer(uint32_t index, id<MTLBuffer> buffer, uint32_t offset)
{
    if (_encoder && _buffer) {
        [_encoder setBuffer:buffer offset:offset atIndex:index];
    }
}

void MetalArgBuffer::setTexture(uint32_t index, id<MTLTexture> texture)
{
    if (_encoder && _buffer) {
        [_encoder setTexture:texture atIndex:index];
    }
}

void MetalArgBuffer::setSamplerState(uint32_t index, id<MTLSamplerState> sampler)
{
    if (_encoder && _buffer) {
        [_encoder setSamplerState:sampler atIndex:index];
    }
}

void MetalArgBuffer::setAccelerationStructure(uint32_t index, id<MTLAccelerationStructure> accelStructure)
{
    if (_encoder && _buffer) {
        if (@available(macOS 12.0, *)) {
            [_encoder setAccelerationStructure:accelStructure atIndex:index];
        } else {
            std::cerr << "Warning: Acceleration structures are only available on macOS 12.0 and later" << std::endl;
        }
    }
}

NSUInteger MetalArgBuffer::encodedLength() const
{
    if (_encoder) {
        return [_encoder encodedLength];
    }
    return 0;
}

// MetalArgBufferManager implementation
MetalArgBufferManager::MetalArgBufferManager(MetalContext* context)
    : _context(context)
{
}

MetalArgBufferManager::~MetalArgBufferManager()
{
    clearArgBuffers();
}

bool MetalArgBufferManager::initialize()
{
    if (!_context) {
        return false;
    }
    
    // Check if device supports argument buffers
    id<MTLDevice> device = _context->device();
    if (!device) {
        return false;
    }
    
    // Make sure the device supports argument buffers
    if ([device argumentBuffersSupport] < MTLArgumentBuffersTier1) {
        std::cerr << "Warning: Device does not support argument buffers, some features will be disabled" << std::endl;
    }
    
    return true;
}

std::shared_ptr<MetalArgBuffer> MetalArgBufferManager::createArgBuffer(
    const std::string& functionName, 
    ArgBufferType type,
    id<MTLLibrary> library)
{
    // Check if we already have this argument buffer
    auto it = _argBuffers.find(functionName);
    if (it != _argBuffers.end()) {
        return it->second;
    }
    
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nullptr;
    }
    
    // Get the function from the library
    if (!library) {
        library = _context->resources()->defaultLibrary();
    }
    
    if (!library) {
        std::cerr << "MetalArgBufferManager::createArgBuffer: No library available" << std::endl;
        return nullptr;
    }
    
    NSString* nsFunctionName = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:nsFunctionName];
    
    if (!function) {
        std::cerr << "MetalArgBufferManager::createArgBuffer: Function not found: " << functionName << std::endl;
        return nullptr;
    }
    
    // Create the argument encoder from the function
    MTLFunctionConstantValues* constants = nil; // No constants for now
    NSError* error = nil;
    id<MTLArgumentEncoder> encoder = nil;
    
    switch (type) {
        case ArgBufferType::Vertex:
            encoder = [function newArgumentEncoderWithBufferIndex:0];
            break;
        case ArgBufferType::Fragment:
            encoder = [function newArgumentEncoderWithBufferIndex:0];
            break;
        case ArgBufferType::Compute:
            encoder = [function newArgumentEncoderWithBufferIndex:0];
            break;
        case ArgBufferType::RayData:
            if (@available(macOS 12.0, *)) {
                encoder = [function newArgumentEncoderWithBufferIndex:0];
            } else {
                std::cerr << "MetalArgBufferManager::createArgBuffer: Ray tracing is only available on macOS 12.0 and later" << std::endl;
            }
            break;
    }
    
    [function release];
    
    if (!encoder) {
        std::cerr << "MetalArgBufferManager::createArgBuffer: Failed to create argument encoder for: " << functionName << std::endl;
        return nullptr;
    }
    
    // Create the argument buffer
    auto argBuffer = std::make_shared<MetalArgBuffer>(device, encoder, functionName);
    [encoder release]; // argBuffer retained the encoder
    
    // Cache the argument buffer
    _argBuffers[functionName] = argBuffer;
    
    return argBuffer;
}

std::shared_ptr<MetalArgBuffer> MetalArgBufferManager::createArgBufferFromLayout(
    const std::vector<MTLArgumentDescriptor*>& descriptors,
    const std::string& name)
{
    // Check if we already have this argument buffer
    auto it = _argBuffers.find(name);
    if (it != _argBuffers.end()) {
        return it->second;
    }
    
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nullptr;
    }
    
    // Convert vector to NSArray
    NSMutableArray* descriptorArray = [NSMutableArray arrayWithCapacity:descriptors.size()];
    for (MTLArgumentDescriptor* descriptor : descriptors) {
        [descriptorArray addObject:descriptor];
    }
    
    // Create the argument encoder from the descriptors
    id<MTLArgumentEncoder> encoder = [device newArgumentEncoderWithArguments:descriptorArray];
    
    if (!encoder) {
        std::cerr << "MetalArgBufferManager::createArgBufferFromLayout: Failed to create argument encoder for: " << name << std::endl;
        return nullptr;
    }
    
    // Create the argument buffer
    auto argBuffer = std::make_shared<MetalArgBuffer>(device, encoder, name);
    [encoder release]; // argBuffer retained the encoder
    
    // Cache the argument buffer
    _argBuffers[name] = argBuffer;
    
    return argBuffer;
}

std::shared_ptr<MetalArgBuffer> MetalArgBufferManager::getArgBuffer(const std::string& name) const
{
    auto it = _argBuffers.find(name);
    if (it != _argBuffers.end()) {
        return it->second;
    }
    return nullptr;
}

void MetalArgBufferManager::clearArgBuffers()
{
    _argBuffers.clear();
}

} // namespace graphics_metal
} // namespace chimerax
