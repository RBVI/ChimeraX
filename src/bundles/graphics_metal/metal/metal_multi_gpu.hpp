// metal_multi_gpu.hpp
// Multi-GPU coordination for Metal rendering in ChimeraX

#pragma once

#import <Metal/Metal.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace chimerax {
namespace graphics_metal {

// Forward declarations
class MetalContext;
class MetalEventManager;

/**
 * Multi-GPU rendering strategies
 */
enum class MultiGPUStrategy {
    SplitFrame = 0,    // Each GPU renders a portion of the screen
    TaskBased = 1,     // Distribute rendering tasks across GPUs
    Alternating = 2,   // Alternate frames between GPUs
    ComputeOffload = 3 // Main GPU renders, other GPUs handle compute tasks
};

/**
 * Information about a GPU device
 */
struct GPUDeviceInfo {
    std::string name;
    bool isPrimary;
    bool isActive;
    bool unifiedMemory;
    uint64_t memorySize;
};

/**
 * Manages synchronization between multiple GPUs
 */
class MetalMultiGPU {
public:
    MetalMultiGPU();
    ~MetalMultiGPU();
    
    // Initialize with Metal context
    bool initialize(MetalContext* context);
    
    // Get available devices
    std::vector<GPUDeviceInfo> getDeviceInfo() const;
    
    // Enable/disable multi-GPU and set strategy
    bool enable(bool enabled, MultiGPUStrategy strategy = MultiGPUStrategy::SplitFrame);
    bool isEnabled() const { return _enabled; }
    
    // Get current strategy
    MultiGPUStrategy getStrategy() const { return _strategy; }
    
    // Active device management
    bool setDeviceActive(id<MTLDevice> device, bool active);
    bool isDeviceActive(id<MTLDevice> device) const;
    std::vector<id<MTLDevice>> getActiveDevices() const;
    
    // Workload distribution
    void computeSplitFrameRegions(uint32_t width, uint32_t height, std::vector<MTLRegion>& regions);
    
    // Synchronization
    id<MTLEvent> signalEvent(id<MTLCommandBuffer> commandBuffer, id<MTLDevice> waitingDevice);
    void waitForEvent(id<MTLCommandBuffer> commandBuffer, id<MTLEvent> event, uint64_t value);
    
    // Resource sharing
    bool shareResource(id<MTLResource> resource, id<MTLDevice> sourceDevice, id<MTLDevice> targetDevice);
    
    // Frame pacing for alternating strategy
    void beginFrame();
    void endFrame();
    id<MTLDevice> getCurrentFrameDevice() const;
    
private:
    MetalContext* _context;
    bool _enabled;
    MultiGPUStrategy _strategy;
    std::vector<id<MTLDevice>> _activeDevices;
    std::unordered_map<id<MTLDevice>, bool> _deviceActivityMap;
    
    // Frame counter for alternating strategy
    uint64_t _frameCounter;
    
    // Cached event manager for synchronization
    MetalEventManager* _eventManager;
    
    // Helper method to calculate device workload ratios
    std::vector<float> calculateDeviceWorkloadRatios() const;
};

} // namespace graphics_metal
} // namespace chimerax
