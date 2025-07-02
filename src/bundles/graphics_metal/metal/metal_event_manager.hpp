// metal_event_manager.hpp
// Manages event synchronization for multi-GPU rendering

#pragma once

#import <Metal/Metal.h>
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <mutex>

namespace chimerax {
namespace graphics_metal {

// Forward declarations
class MetalContext;

/**
 * Manages events for GPU synchronization
 * 
 * Events are used to coordinate work between multiple GPUs, ensuring
 * that resources are accessed safely and work is properly sequenced.
 */
class MetalEventManager {
public:
    MetalEventManager(MetalContext* context);
    ~MetalEventManager();
    
    // Initialization
    bool initialize();
    
    // Create a new named event
    id<MTLEvent> createEvent(const std::string& name = "");
    
    // Get an existing event by name
    id<MTLEvent> getEvent(const std::string& name) const;
    
    // Signal an event at the end of a command buffer
    id<MTLEvent> signalEvent(id<MTLCommandBuffer> commandBuffer, id<MTLDevice> targetDevice = nil);
    
    // Wait for an event to reach a specific value
    void waitForEvent(id<MTLCommandBuffer> commandBuffer, id<MTLEvent> event, uint64_t value);
    
    // Event barriers - ensure all devices have completed work
    void syncAllDevices();
    void beginFrameBarrier();
    void endFrameBarrier();
    
    // Cleanup
    void releaseEvents();
    
private:
    MetalContext* _context;
    
    // Named events
    std::unordered_map<std::string, id<MTLEvent>> _namedEvents;
    
    // Special events for common synchronization points
    id<MTLEvent> _frameStartEvent;
    id<MTLEvent> _frameEndEvent;
    
    // Current event values
    std::unordered_map<id<MTLEvent>, uint64_t> _eventValues;
    
    // Mutex for thread safety
    mutable std::mutex _mutex;
    
    // Internal methods
    id<MTLEvent> createEventForDevice(id<MTLDevice> device);
};

} // namespace graphics_metal
} // namespace chimerax
