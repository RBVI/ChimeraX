// metal_multi_gpu.cpp
// Multi-device coordination for Metal in ChimeraX.

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
    , _strategy(MultiGPUStrategy::DeviceSelection)
    , _presentationDevice(nil)
    , _computeDevice(nil)
    , _computeQueue(nil)
    , _eventManager(nullptr)
{
}

MetalMultiGPU::~MetalMultiGPU()
{
    // Owned objects are released by ARC; we don't own _context or _eventManager.
}

bool MetalMultiGPU::initialize(MetalContext* context)
{
    if (!context) return false;
    _context = context;
    _eventManager = _context->eventManager();
    _allDevices = _context->allDevices();
    // Start with the primary device as the presentation device.
    _presentationDevice = _context->device();
    return true;
}

// ---------------------------------------------------------------------------
// Device inventory
// ---------------------------------------------------------------------------

std::vector<GPUDeviceInfo> MetalMultiGPU::getDeviceInfo() const
{
    std::vector<GPUDeviceInfo> infos;
    if (!_context) return infos;

    id<MTLDevice> primary = _context->device();
    for (id<MTLDevice> dev : _allDevices) {
        GPUDeviceInfo info;
        info.name         = [[dev name] UTF8String];
        info.isPrimary    = (dev == primary);
        info.isActive     = (dev == _presentationDevice || dev == _computeDevice);
        info.unifiedMemory = [dev hasUnifiedMemory];
        info.isLowPower   = [dev isLowPower];
        if (@available(macOS 10.13, *)) {
            info.memorySize = [dev recommendedMaxWorkingSetSize];
        } else {
            info.memorySize = 0;
        }
        infos.push_back(info);
    }
    return infos;
}

// ---------------------------------------------------------------------------
// Enable / strategy
// ---------------------------------------------------------------------------

bool MetalMultiGPU::enable(bool enabled, MultiGPUStrategy strategy)
{
    switch (strategy) {
        case MultiGPUStrategy::SplitFrame:
        case MultiGPUStrategy::TaskBased:
        case MultiGPUStrategy::Alternating:
            _logUnsupportedStrategy(strategy);
            // Fall through to DeviceSelection as the safe default.
            strategy = MultiGPUStrategy::DeviceSelection;
            break;
        default:
            break;
    }

    _enabled  = enabled;
    _strategy = strategy;

    if (!enabled) {
        // Revert to primary device for both roles.
        _presentationDevice = _context ? _context->device() : nil;
        _computeDevice      = nil;
        _computeQueue       = nil;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Device selection
// ---------------------------------------------------------------------------

bool MetalMultiGPU::selectPresentationDevice(const std::string& deviceName)
{
    if (deviceName.empty()) {
        _presentationDevice = _context ? _context->device() : nil;
        return true;
    }
    id<MTLDevice> dev = _findDeviceByName(deviceName);
    if (!dev) {
        std::cerr << "MetalMultiGPU: no device named '" << deviceName << "'" << std::endl;
        return false;
    }
    _presentationDevice = dev;
    return true;
}

bool MetalMultiGPU::selectComputeDevice(const std::string& deviceName)
{
    if (deviceName.empty()) {
        _computeDevice = nil;
        _computeQueue  = nil;
        return true;
    }
    id<MTLDevice> dev = _findDeviceByName(deviceName);
    if (!dev) {
        std::cerr << "MetalMultiGPU: no device named '" << deviceName << "'" << std::endl;
        return false;
    }
    _computeDevice = dev;
    _computeQueue  = [dev newCommandQueue];
    if (!_computeQueue) {
        std::cerr << "MetalMultiGPU: failed to create command queue on '"
                  << deviceName << "'" << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Compute offload
// ---------------------------------------------------------------------------

id<MTLSharedEvent> MetalMultiGPU::submitComputeWork(
    id<MTLCommandBuffer> computeCommandBuffer,
    id<MTLDevice> fromDevice)
{
    if (!_computeDevice || !_computeQueue) return nil;

    // Signal a shared event from the compute command buffer so the caller can
    // synchronise on it from the primary device's command buffer.
    id<MTLSharedEvent> sharedEvent = [_computeDevice newSharedEvent];
    if (!sharedEvent) return nil;

    static uint64_t signalValue = 1;
    [computeCommandBuffer encodeSignalEvent:sharedEvent value:signalValue];
    [computeCommandBuffer commit];
    signalValue++;
    return sharedEvent;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

id<MTLDevice> MetalMultiGPU::_findDeviceByName(const std::string& name) const
{
    for (id<MTLDevice> dev : _allDevices) {
        if ([[dev name] UTF8String] == name) return dev;
    }
    return nil;
}

void MetalMultiGPU::_logUnsupportedStrategy(MultiGPUStrategy strategy) const
{
    const char* sname = "unknown";
    switch (strategy) {
        case MultiGPUStrategy::SplitFrame:  sname = "SplitFrame";  break;
        case MultiGPUStrategy::TaskBased:   sname = "TaskBased";   break;
        case MultiGPUStrategy::Alternating: sname = "Alternating"; break;
        default: break;
    }
    std::cerr << "MetalMultiGPU: strategy '" << sname
              << "' is not supported on macOS Metal (no SLI / multi-head rendering). "
              << "Falling back to DeviceSelection." << std::endl;
}

} // namespace graphics_metal
} // namespace chimerax
