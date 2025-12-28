// metal_context.hpp
// Optimized Metal context management for ChimeraX with multi-GPU support

#pragma once

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace chimerax {
namespace graphics_metal {

// Forward declarations
class MetalResources;
class MetalHeapManager;
class MetalArgBufferManager;
class MetalEventManager;

/**
 * Device capability structure to track feature support
 */
struct DeviceCapabilities {
    bool unifiedMemory;
    bool rayTracing;
    bool rasterOrderGroups;
    bool meshShaders;
    bool argumentBuffers;
    bool indirectCommandBuffers;
    bool familyApple1;
    bool familyApple2;
    bool familyApple3;
    bool familyApple4;
    bool familyApple5;
    bool familyApple6;
    bool familyApple7;
    bool familyMac1;
    bool familyMac2;
};

/**
 * Manages Metal device and context for ChimeraX graphics rendering
 * Enhanced with multi-GPU support and advanced Metal features
 */
class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    // Initialization
    bool initialize();
    bool isInitialized() const { return _initialized; }
    std::string getErrorMessage() const { return _errorMessage; }
    
    // Device access
    id<MTLDevice> device() const { return _device; }
    id<MTLCommandQueue> commandQueue() const { return _commandQueue; }
    
    // Multiple device support
    bool hasMultipleDevices() const { return _allDevices.size() > 1; }
    std::vector<id<MTLDevice>> allDevices() const { return _allDevices; }
    std::vector<id<MTLDevice>> activeDevices() const;
    id<MTLDevice> deviceAtIndex(size_t index) const;
    id<MTLCommandQueue> commandQueueForDevice(id<MTLDevice> device) const;
    
    // Device info
    std::string deviceName() const;
    std::string deviceVendor() const;
    DeviceCapabilities deviceCapabilities() const { return _capabilities; }
    
    // Resource management
    MetalResources* resources() const { return _resources.get(); }
    MetalHeapManager* heapManager() const { return _heapManager.get(); }
    MetalArgBufferManager* argBufferManager() const { return _argBufferManager.get(); }
    MetalEventManager* eventManager() const { return _eventManager.get(); }
    
    // View management
    void setDrawableSize(CGSize size);
    CGSize drawableSize() const { return _drawableSize; }
    
    // Render target
    MTKView* mtkView() const { return _mtkView; }
    void setMTKView(MTKView* view);
    
    // Device capabilities
    bool supportsUnifiedMemory() const { return _capabilities.unifiedMemory; }
    bool supportsRayTracing() const { return _capabilities.rayTracing; }
    bool supportsMeshShaders() const { return _capabilities.meshShaders; }
    bool supportsArgumentBuffers() const { return _capabilities.argumentBuffers; }
    bool supportsIndirectCommandBuffers() const { return _capabilities.indirectCommandBuffers; }
    
    // Label setting for debugging
    void setLabel(id<MTLResource> resource, const std::string& label);
    
    // Debug capture
    void beginCapture();
    void endCapture();
    
private:
    // Core Metal objects
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    MTKView* _mtkView;
    
    // Multi-GPU support
    std::vector<id<MTLDevice>> _allDevices;
    std::unordered_map<id<MTLDevice>, id<MTLCommandQueue>> _deviceCommandQueues;
    
    // Resource managers
    std::unique_ptr<MetalResources> _resources;
    std::unique_ptr<MetalHeapManager> _heapManager;
    std::unique_ptr<MetalArgBufferManager> _argBufferManager;
    std::unique_ptr<MetalEventManager> _eventManager;
    
    // State tracking
    bool _initialized;
    std::string _errorMessage;
    CGSize _drawableSize;
    DeviceCapabilities _capabilities;
    
    // Internal methods
    bool discoverDevices();
    bool createDeviceCommandQueues();
    bool initializeResourceManagers();
    void detectDeviceCapabilities();
};

} // namespace graphics_metal
} // namespace chimerax
