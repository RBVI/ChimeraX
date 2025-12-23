// metal_context.cpp
// Optimized implementation of Metal context management for ChimeraX

#include "metal_context.hpp"
#include "metal_resources.hpp"
#include "metal_heap_manager.hpp"
#include "metal_argbuffer_manager.hpp"
#include "metal_event_manager.hpp"
#include <iostream>
#include <algorithm>

namespace chimerax {
namespace graphics_metal {

MetalContext::MetalContext()
    : _device(nil)
    , _commandQueue(nil)
    , _mtkView(nil)
    , _resources(nullptr)
    , _heapManager(nullptr)
    , _argBufferManager(nullptr)
    , _eventManager(nullptr)
    , _initialized(false)
    , _drawableSize(CGSizeMake(0, 0))
{
    // Initialize capabilities to false
    memset(&_capabilities, 0, sizeof(DeviceCapabilities));
}

MetalContext::~MetalContext()
{
    // Resources and managers must be released before devices
    _resources.reset();
    _heapManager.reset();
    _argBufferManager.reset();
    _eventManager.reset();
    
    // Release command queues
    for (auto& pair : _deviceCommandQueues) {
        if (pair.second) {
            [pair.second release];
        }
    }
    _deviceCommandQueues.clear();
    
    // Release devices
    for (auto& device : _allDevices) {
        if (device && device != _device) { // Don't double-release _device
            [device release];
        }
    }
    _allDevices.clear();
    
    // Release primary device and queue
    if (_commandQueue) {
        [_commandQueue release];
        _commandQueue = nil;
    }
    
    if (_device) {
        [_device release];
        _device = nil;
    }
    
    // Note: We don't own the MTKView, so we don't release it
}

bool MetalContext::initialize()
{
    if (_initialized) {
        return true;
    }
    
    // Discover available Metal devices
    if (!discoverDevices()) {
        return false;
    }
    
    // Create command queues for all devices
    if (!createDeviceCommandQueues()) {
        return false;
    }
    
    // Detect device capabilities
    detectDeviceCapabilities();
    
    // Initialize resource managers
    if (!initializeResourceManagers()) {
        return false;
    }
    
    _initialized = true;
    return true;
}

bool MetalContext::discoverDevices()
{
    // Get all available Metal devices
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if ([devices count] == 0) {
        _errorMessage = "No Metal devices found";
        [devices release];
        return false;
    }
    
    // Find the most suitable device for primary
    // Prefer discrete GPUs if available
    id<MTLDevice> discreteGPU = nil;
    id<MTLDevice> integratedGPU = nil;
    
    for (id<MTLDevice> device in devices) {
        if ([device isLowPower]) {
            if (!integratedGPU) {
                integratedGPU = device;
            }
        } else {
            if (!discreteGPU) {
                discreteGPU = device;
            }
        }
        
        // Add to all devices list
        [device retain];
        _allDevices.push_back(device);
    }
    
    // Select primary device - prefer discrete GPU if available
    if (discreteGPU) {
        _device = discreteGPU;
    } else if (integratedGPU) {
        _device = integratedGPU;
    } else {
        // Default to first device
        _device = [devices objectAtIndex:0];
    }
    
    // Retain primary device
    [_device retain];
    
    [devices release];
    return true;
}

bool MetalContext::createDeviceCommandQueues()
{
    // Create primary command queue
    _commandQueue = [_device newCommandQueue];
    if (!_commandQueue) {
        _errorMessage = "Failed to create primary Metal command queue";
        return false;
    }
    
    [_commandQueue setLabel:@"ChimeraX Primary Command Queue"];
    _deviceCommandQueues[_device] = _commandQueue;
    
    // Create command queues for all other devices
    for (auto& device : _allDevices) {
        if (device != _device) {
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) {
                std::cerr << "Warning: Failed to create command queue for secondary device" << std::endl;
                continue;
            }
            
            [queue setLabel:@"ChimeraX Secondary Command Queue"];
            _deviceCommandQueues[device] = queue;
        }
    }
    
    return true;
}

bool MetalContext::initializeResourceManagers()
{
    // Create resource manager
    _resources = std::make_unique<MetalResources>(this);
    if (!_resources->initialize()) {
        _errorMessage = "Failed to initialize Metal resources";
        return false;
    }
    
    // Create heap manager (for efficient memory allocation)
    _heapManager = std::make_unique<MetalHeapManager>(this);
    if (!_heapManager->initialize()) {
        _errorMessage = "Failed to initialize Metal heap manager";
        return false;
    }
    
    // Create argument buffer manager
    _argBufferManager = std::make_unique<MetalArgBufferManager>(this);
    if (!_argBufferManager->initialize()) {
        _errorMessage = "Failed to initialize Metal argument buffer manager";
        return false;
    }
    
    // Create event manager for multi-GPU synchronization
    _eventManager = std::make_unique<MetalEventManager>(this);
    if (!_eventManager->initialize()) {
        _errorMessage = "Failed to initialize Metal event manager";
        return false;
    }
    
    return true;
}

void MetalContext::detectDeviceCapabilities()
{
    if (!_device) {
        return;
    }
    
    // Check for unified memory (Apple Silicon)
    _capabilities.unifiedMemory = [_device hasUnifiedMemory];
    
    // Check for ray tracing support (Metal 3)
    if (@available(macOS 12.0, *)) {
        _capabilities.rayTracing = [_device supportsRaytracing];
    } else {
        _capabilities.rayTracing = false;
    }
    
    // Check for mesh shader support (Metal 3)
    if (@available(macOS 13.0, *)) {
        _capabilities.meshShaders = [_device supportsFamilyMac2] || 
                                     [_device supportsFamilyApple7];
    } else {
        _capabilities.meshShaders = false;
    }
    
    // Check for advanced argument buffer support
    _capabilities.argumentBuffers = [_device argumentBuffersSupport] >= MTLArgumentBuffersTier2;
    
    // Check for indirect command buffer support
    if (@available(macOS 11.0, *)) {
        _capabilities.indirectCommandBuffers = [_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1];
    } else {
        _capabilities.indirectCommandBuffers = false;
    }
    
    // Check for raster order groups (important for deferred rendering)
    _capabilities.rasterOrderGroups = [_device supportsRasterizationRateMapWithLayerCount:1];
    
    // GPU Family support
    _capabilities.familyApple1 = [_device supportsFamily:MTLGPUFamilyApple1];
    _capabilities.familyApple2 = [_device supportsFamily:MTLGPUFamilyApple2];
    _capabilities.familyApple3 = [_device supportsFamily:MTLGPUFamilyApple3];
    _capabilities.familyApple4 = [_device supportsFamily:MTLGPUFamilyApple4];
    _capabilities.familyApple5 = [_device supportsFamily:MTLGPUFamilyApple5];
    
    // Check for Apple6 and Apple7 (newer families)
    if (@available(macOS 12.0, *)) {
        _capabilities.familyApple6 = [_device supportsFamily:MTLGPUFamilyApple6];
    } else {
        _capabilities.familyApple6 = false;
    }
    
    if (@available(macOS 13.0, *)) {
        _capabilities.familyApple7 = [_device supportsFamily:MTLGPUFamilyApple7];
    } else {
        _capabilities.familyApple7 = false;
    }
    
    // Mac GPU families
    _capabilities.familyMac1 = [_device supportsFamily:MTLGPUFamilyMac1];
    _capabilities.familyMac2 = [_device supportsFamily:MTLGPUFamilyMac2];
}

void MetalContext::setMTKView(MTKView* view)
{
    _mtkView = view;
    
    if (_mtkView) {
        // Configure the view to use our device
        [_mtkView setDevice:_device];
        [_mtkView setColorPixelFormat:MTLPixelFormatBGRA8Unorm];
        [_mtkView setDepthStencilPixelFormat:MTLPixelFormatDepth32Float];
        [_mtkView setSampleCount:1]; // Start with no MSAA, can be adjusted later
        
        // Update drawable size
        setDrawableSize([_mtkView drawableSize]);
    }
}

void MetalContext::setDrawableSize(CGSize size)
{
    _drawableSize = size;
}

std::vector<id<MTLDevice>> MetalContext::activeDevices() const
{
    return _allDevices;
}

id<MTLDevice> MetalContext::deviceAtIndex(size_t index) const
{
    if (index < _allDevices.size()) {
        return _allDevices[index];
    }
    return nil;
}

id<MTLCommandQueue> MetalContext::commandQueueForDevice(id<MTLDevice> device) const
{
    auto it = _deviceCommandQueues.find(device);
    if (it != _deviceCommandQueues.end()) {
        return it->second;
    }
    return nil;
}

std::string MetalContext::deviceName() const
{
    if (!_device) {
        return "Unknown";
    }
    
    return [[_device name] UTF8String];
}

std::string MetalContext::deviceVendor() const
{
    if (!_device) {
        return "Unknown";
    }
    
    // There's no direct vendor info in Metal, so infer from device name
    std::string name = [[_device name] UTF8String];
    if (name.find("AMD") != std::string::npos) {
        return "AMD";
    } else if (name.find("NVIDIA") != std::string::npos) {
        return "NVIDIA";
    } else if (name.find("Intel") != std::string::npos) {
        return "Intel";
    } else if (name.find("Apple") != std::string::npos) {
        return "Apple";
    }
    
    return "Unknown";
}

void MetalContext::setLabel(id<MTLResource> resource, const std::string& label)
{
    if (resource) {
        NSString* nsLabel = [NSString stringWithUTF8String:label.c_str()];
        [resource setLabel:nsLabel];
    }
}

void MetalContext::beginCapture()
{
    MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor* captureDescriptor = [[MTLCaptureDescriptor alloc] init];
    captureDescriptor.captureObject = _device;
    
    NSError* error = nil;
    if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
        std::cerr << "Failed to start Metal capture: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    
    [captureDescriptor release];
}

void MetalContext::endCapture()
{
    MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
    [captureManager stopCapture];
}

} // namespace graphics_metal
} // namespace chimerax
