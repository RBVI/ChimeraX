// metal_event_manager.cpp
// Implementation of event synchronization for multi-GPU rendering

#include "metal_event_manager.hpp"
#include "metal_context.hpp"
#include <iostream>

namespace chimerax {
namespace graphics_metal {

MetalEventManager::MetalEventManager(MetalContext* context)
    : _context(context)
    , _frameStartEvent(nil)
    , _frameEndEvent(nil)
{
}

MetalEventManager::~MetalEventManager()
{
    releaseEvents();
}

bool MetalEventManager::initialize()
{
    if (!_context) {
        return false;
    }
    
    // Check if Metal device supports shared events
    id<MTLDevice> device = _context->device();
    if (!device) {
        return false;
    }
    
    // Create special events for frame synchronization
    _frameStartEvent = createEventForDevice(device);
    if (!_frameStartEvent) {
        std::cerr << "MetalEventManager::initialize: Failed to create frame start event" << std::endl;
        return false;
    }
    
    _frameEndEvent = createEventForDevice(device);
    if (!_frameEndEvent) {
        std::cerr << "MetalEventManager::initialize: Failed to create frame end event" << std::endl;
        [_frameStartEvent release];
        _frameStartEvent = nil;
        return false;
    }
    
    // Set initial values
    _eventValues[_frameStartEvent] = 0;
    _eventValues[_frameEndEvent] = 0;
    
    // Label events for debugging
    [_frameStartEvent setLabel:@"Frame Start Event"];
    [_frameEndEvent setLabel:@"Frame End Event"];
    
    return true;
}

id<MTLEvent> MetalEventManager::createEvent(const std::string& name)
{
    std::lock_guard<std::mutex> lock(_mutex);
    
    // If name is provided, check if it already exists
    if (!name.empty()) {
        auto it = _namedEvents.find(name);
        if (it != _namedEvents.end()) {
            return it->second;
        }
    }
    
    // Create new event
    id<MTLDevice> device = _context->device();
    if (!device) {
        return nil;
    }
    
    id<MTLEvent> event = createEventForDevice(device);
    if (!event) {
        return nil;
    }
    
    // Set initial value
    _eventValues[event] = 0;
    
    // Add to named events if name provided
    if (!name.empty()) {
        [event setLabel:[NSString stringWithUTF8String:name.c_str()]];
        _namedEvents[name] = event;
    }
    
    return event;
}

id<MTLEvent> MetalEventManager::getEvent(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    
    auto it = _namedEvents.find(name);
    if (it != _namedEvents.end()) {
        return it->second;
    }
    
    return nil;
}

id<MTLEvent> MetalEventManager::signalEvent(id<MTLCommandBuffer> commandBuffer, id<MTLDevice> targetDevice)
{
    if (!commandBuffer) {
        return nil;
    }
    
    // Determine which event to use
    id<MTLEvent> event = nil;
    
    if (targetDevice) {
        // Create a new event for the specific target device
        event = createEventForDevice(targetDevice);
        if (!event) {
            std::cerr << "MetalEventManager::signalEvent: Failed to create event for target device" << std::endl;
            return nil;
        }
        
        // Initialize event value
        std::lock_guard<std::mutex> lock(_mutex);
        _eventValues[event] = 0;
    } else {
        // Use frame end event
        event = _frameEndEvent;
    }
    
    if (!event) {
        return nil;
    }
    
    // Get the next value for this event
    uint64_t value;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        value = ++_eventValues[event];
    }
    
    // Signal the event at the end of the command buffer
    [commandBuffer encodeSignalEvent:event value:value];
    
    return event;
}

void MetalEventManager::waitForEvent(id<MTLCommandBuffer> commandBuffer, id<MTLEvent> event, uint64_t value)
{
    if (!commandBuffer || !event) {
        return;
    }
    
    // Encode a wait for the event
    [commandBuffer encodeWaitForEvent:event value:value];
}

void MetalEventManager::syncAllDevices()
{
    if (!_context) {
        return;
    }
    
    // Get all active devices
    std::vector<id<MTLDevice>> devices = _context->allDevices();
    if (devices.empty()) {
        return;
    }
    
    // Signal an event on each device and have all other devices wait for it
    for (id<MTLDevice> device : devices) {
        id<MTLCommandQueue> queue = _context->commandQueueForDevice(device);
        if (!queue) {
            continue;
        }
        
        // Create a command buffer
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        if (!cmdBuffer) {
            continue;
        }
        
        // Signal an event
        id<MTLEvent> event = createEventForDevice(device);
        if (!event) {
            [cmdBuffer release];
            continue;
        }
        
        uint64_t value;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            value = ++_eventValues[event];
        }
        
        [cmdBuffer encodeSignalEvent:event value:value];
        [cmdBuffer commit];
        
        // Have all other devices wait for this event
        for (id<MTLDevice> waitDevice : devices) {
            if (waitDevice == device) {
                continue;
            }
            
            id<MTLCommandQueue> waitQueue = _context->commandQueueForDevice(waitDevice);
            if (!waitQueue) {
                continue;
            }
            
            id<MTLCommandBuffer> waitCmdBuffer = [waitQueue commandBuffer];
            if (!waitCmdBuffer) {
                continue;
            }
            
            [waitCmdBuffer encodeWaitForEvent:event value:value];
            [waitCmdBuffer commit];
            [waitCmdBuffer waitUntilCompleted];
            [waitCmdBuffer release];
        }
        
        [cmdBuffer waitUntilCompleted];
        [cmdBuffer release];
        [event release];
    }
}

void MetalEventManager::beginFrameBarrier()
{
    if (!_context || !_frameStartEvent) {
        return;
    }
    
    // Get the command queue for the primary device
    id<MTLCommandQueue> queue = _context->commandQueue();
    if (!queue) {
        return;
    }
    
    // Create a command buffer
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    if (!cmdBuffer) {
        return;
    }
    
    // Signal the frame start event
    uint64_t value;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        value = ++_eventValues[_frameStartEvent];
    }
    
    [cmdBuffer encodeSignalEvent:_frameStartEvent value:value];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
    [cmdBuffer release];
}

void MetalEventManager::endFrameBarrier()
{
    if (!_context || !_frameEndEvent) {
        return;
    }
    
    // Get the command queue for the primary device
    id<MTLCommandQueue> queue = _context->commandQueue();
    if (!queue) {
        return;
    }
    
    // Create a command buffer
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    if (!cmdBuffer) {
        return;
    }
    
    // Signal the frame end event
    uint64_t value;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        value = ++_eventValues[_frameEndEvent];
    }
    
    [cmdBuffer encodeSignalEvent:_frameEndEvent value:value];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
    [cmdBuffer release];
}

void MetalEventManager::releaseEvents()
{
    std::lock_guard<std::mutex> lock(_mutex);
    
    // Release special events
    if (_frameStartEvent) {
        [_frameStartEvent release];
        _frameStartEvent = nil;
    }
    
    if (_frameEndEvent) {
        [_frameEndEvent release];
        _frameEndEvent = nil;
    }
    
    // Release named events
    for (auto& pair : _namedEvents) {
        [pair.second release];
    }
    _namedEvents.clear();
    
    // Clear event values
    _eventValues.clear();
}

id<MTLEvent> MetalEventManager::createEventForDevice(id<MTLDevice> device)
{
    if (!device) {
        return nil;
    }
    
    // Create a new event
    id<MTLEvent> event = [device newEvent];
    if (!event) {
        std::cerr << "MetalEventManager::createEventForDevice: Failed to create event" << std::endl;
    }
    
    return event;
}

} // namespace graphics_metal
} // namespace chimerax
