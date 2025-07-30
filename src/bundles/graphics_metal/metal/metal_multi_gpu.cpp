// metal_multi_gpu.cpp
// Implementation of multi-GPU coordination for Metal rendering in ChimeraX

#include "metal_multi_gpu.hpp"
#include "metal_context.hpp"
#include "metal_event_manager.hpp"
#include <iostream>
#include <algorithm>

namespace chimerax {
namespace graphics_metal {

MetalMultiGPU::MetalMultiGPU()
    : _context(nullptr)
    , _enabled(false)
    , _strategy(MultiGPUStrategy::SplitFrame)
    , _frameCounter(0)
    , _eventManager(nullptr)
{
}

MetalMultiGPU::~MetalMultiGPU()
{
    // We don't own _context or _eventManager, so no need to delete them
}

bool MetalMultiGPU::initialize(MetalContext* context)
{
    if (!context) {
        return false;
    }
    
    _context = context;
    
    // Get event manager from context
    _eventManager = _context->eventManager();
    if (!_eventManager) {
        std::cerr << "MetalMultiGPU::initialize: No event manager available" << std::endl;
        return false;
    }
    
    // Check if multiple devices are available
    std::vector<id<MTLDevice>> devices = _context->allDevices();
    if (devices.size() <= 1) {
        std::cerr << "MetalMultiGPU::initialize: Only one device available, multi-GPU disabled" << std::endl;
        return true; // Not an error, just no multi-GPU
    }
    
    // Initialize device activity map - start with all devices active
    for (id<MTLDevice> device : devices) {
        _deviceActivityMap[device] = true;
        _activeDevices.push_back(device);
    }
    
    return true;
}

std::vector<GPUDeviceInfo> MetalMultiGPU::getDeviceInfo() const
{
    std::vector<GPUDeviceInfo> deviceInfos;
    
    if (!_context) {
        return deviceInfos;
    }
    
    // Get all devices from context
    std::vector<id<MTLDevice>> devices = _context->allDevices();
    id<MTLDevice> primaryDevice = _context->device();
    
    for (id<MTLDevice> device : devices) {
        GPUDeviceInfo info;
        info.name = [[device name] UTF8String];
        info.isPrimary = (device == primaryDevice);
        info.isActive = isDeviceActive(device);
        info.unifiedMemory = [device hasUnifiedMemory];
        
        // Get memory size if available
        if (@available(macOS 10.13, *)) {
            info.memorySize = [device recommendedMaxWorkingSetSize];
        } else {
            info.memorySize = 0;
        }
        
        deviceInfos.push_back(info);
    }
    
    return deviceInfos;
}

bool MetalMultiGPU::enable(bool enabled, MultiGPUStrategy strategy)
{
    if (!_context) {
        return false;
    }
    
    // Check if multiple devices are available
    if (_context->allDevices().size() <= 1) {
        _enabled = false;
        return false;
    }
    
    _enabled = enabled;
    
    if (enabled) {
        _strategy = strategy;
        
        // Update active devices based on strategy
        _activeDevices.clear();
        for (const auto& pair : _deviceActivityMap) {
            if (pair.second) {
                _activeDevices.push_back(pair.first);
            }
        }
        
        // Log the strategy
        std::string strategyName;
        switch (_strategy) {
            case MultiGPUStrategy::SplitFrame:
                strategyName = "Split Frame";
                break;
            case MultiGPUStrategy::TaskBased:
                strategyName = "Task Based";
                break;
            case MultiGPUStrategy::Alternating:
                strategyName = "Alternating Frames";
                break;
            case MultiGPUStrategy::ComputeOffload:
                strategyName = "Compute Offload";
                break;
        }
        
        std::cout << "Multi-GPU enabled with strategy: " << strategyName << std::endl;
        std::cout << "Active GPUs: " << _activeDevices.size() << std::endl;
    }
    
    return true;
}

bool MetalMultiGPU::setDeviceActive(id<MTLDevice> device, bool active)
{
    if (!_context) {
        return false;
    }
    
    // Check if device is valid
    std::vector<id<MTLDevice>> allDevices = _context->allDevices();
    auto it = std::find(allDevices.begin(), allDevices.end(), device);
    if (it == allDevices.end()) {
        std::cerr << "MetalMultiGPU::setDeviceActive: Invalid device" << std::endl;
        return false;
    }
    
    // Don't allow deactivating the primary device
    if (device == _context->device() && !active) {
        std::cerr << "MetalMultiGPU::setDeviceActive: Cannot deactivate primary device" << std::endl;
        return false;
    }
    
    // Update device activity map
    _deviceActivityMap[device] = active;
    
    // Update active devices list if enabled
    if (_enabled) {
        _activeDevices.clear();
        for (const auto& pair : _deviceActivityMap) {
            if (pair.second) {
                _activeDevices.push_back(pair.first);
            }
        }
    }
    
    return true;
}

bool MetalMultiGPU::isDeviceActive(id<MTLDevice> device) const
{
    if (!_context) {
        return false;
    }
    
    auto it = _deviceActivityMap.find(device);
    if (it != _deviceActivityMap.end()) {
        return it->second;
    }
    
    return false;
}

std::vector<id<MTLDevice>> MetalMultiGPU::getActiveDevices() const
{
    if (!_enabled) {
        // If multi-GPU is disabled, only return the primary device
        std::vector<id<MTLDevice>> result;
        if (_context) {
            result.push_back(_context->device());
        }
        return result;
    }
    
    return _activeDevices;
}

void MetalMultiGPU::computeSplitFrameRegions(uint32_t width, uint32_t height, std::vector<MTLRegion>& regions)
{
    regions.clear();
    
    if (!_enabled || _activeDevices.empty()) {
        // If multi-GPU is disabled, return full frame region
        MTLRegion fullRegion = MTLRegionMake2D(0, 0, width, height);
        regions.push_back(fullRegion);
        return;
    }
    
    // Get workload ratios for active devices
    std::vector<float> ratios = calculateDeviceWorkloadRatios();
    
    // Calculate regions based on strategy
    switch (_strategy) {
        case MultiGPUStrategy::SplitFrame: {
            // Split the frame horizontally based on device ratios
            float totalHeight = static_cast<float>(height);
            float currentY = 0.0f;
            
            for (size_t i = 0; i < _activeDevices.size(); ++i) {
                float deviceRatio = ratios[i];
                uint32_t regionHeight = static_cast<uint32_t>(totalHeight * deviceRatio);
                
                // Ensure the last region covers the rest of the frame
                if (i == _activeDevices.size() - 1) {
                    regionHeight = height - static_cast<uint32_t>(currentY);
                }
                
                MTLRegion region = MTLRegionMake2D(
                    0, static_cast<uint32_t>(currentY),
                    width, regionHeight
                );
                
                regions.push_back(region);
                currentY += regionHeight;
            }
            break;
        }
        
        case MultiGPUStrategy::TaskBased:
        case MultiGPUStrategy::Alternating:
        case MultiGPUStrategy::ComputeOffload:
            // These strategies don't split the frame, so return full frame region
            MTLRegion fullRegion = MTLRegionMake2D(0, 0, width, height);
            regions.push_back(fullRegion);
            break;
    }
}

id<MTLEvent> MetalMultiGPU::signalEvent(id<MTLCommandBuffer> commandBuffer, id<MTLDevice> waitingDevice)
{
    if (!_enabled || !_eventManager || !commandBuffer) {
        return nil;
    }
    
    return _eventManager->signalEvent(commandBuffer, waitingDevice);
}

void MetalMultiGPU::waitForEvent(id<MTLCommandBuffer> commandBuffer, id<MTLEvent> event, uint64_t value)
{
    if (!_enabled || !_eventManager || !commandBuffer || !event) {
        return;
    }
    
    _eventManager->waitForEvent(commandBuffer, event, value);
}

bool MetalMultiGPU::shareResource(id<MTLResource> resource, id<MTLDevice> sourceDevice, id<MTLDevice> targetDevice)
{
    if (!_enabled || !resource || !sourceDevice || !targetDevice) {
        return false;
    }
    
    // Check if the resource is shareable
    if (!([resource storageMode] == MTLStorageModeShared ||
          [resource storageMode] == MTLStorageModeManaged)) {
        std::cerr << "MetalMultiGPU::shareResource: Resource must be in Shared or Managed storage mode" << std::endl;
        return false;
    }
    
    // Create a shared event to synchronize access
    id<MTLEvent> event = [sourceDevice newEvent];
    if (!event) {
        std::cerr << "MetalMultiGPU::shareResource: Failed to create synchronization event" << std::endl;
        return false;
    }
    
    // In a real implementation, we would use the MTLSharedEvent to synchronize
    // access to the resource between the devices. This would involve:
    // 1. Signaling the event on the source device when done with the resource
    // 2. Waiting for the event on the target device before accessing the resource
    
    // For now, we'll just simulate successful sharing
    [event release];
    return true;
}

void MetalMultiGPU::beginFrame()
{
    if (!_enabled) {
        return;
    }
    
    // For alternating strategy, increment frame counter
    if (_strategy == MultiGPUStrategy::Alternating) {
        _frameCounter++;
    }
}

void MetalMultiGPU::endFrame()
{
    // Nothing special to do at end of frame yet
}

id<MTLDevice> MetalMultiGPU::getCurrentFrameDevice() const
{
    if (!_enabled || _activeDevices.empty()) {
        if (_context) {
            return _context->device();
        }
        return nil;
    }
    
    // For alternating strategy, select device based on frame counter
    if (_strategy == MultiGPUStrategy::Alternating) {
        size_t deviceIndex = _frameCounter % _activeDevices.size();
        return _activeDevices[deviceIndex];
    }
    
    // For other strategies, return primary device
    return _context->device();
}

std::vector<float> MetalMultiGPU::calculateDeviceWorkloadRatios() const
{
    std::vector<float> ratios;
    
    if (_activeDevices.empty()) {
        return ratios;
    }
    
    // For now, simply divide work equally among devices
    float equalRatio = 1.0f / static_cast<float>(_activeDevices.size());
    for (size_t i = 0; i < _activeDevices.size(); ++i) {
        ratios.push_back(equalRatio);
    }
    
    // In a more advanced implementation, we could assign workload based on:
    // - Device performance (compute units, memory bandwidth)
    // - Device type (integrated vs discrete)
    // - Historical performance data
    
    return ratios;
}

} // namespace graphics_metal
} // namespace chimerax
