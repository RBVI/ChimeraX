// metal_multi_gpu.hpp
// Multi-device coordination for Metal in ChimeraX.
//
// Metal on macOS does NOT support split-frame SLI-style rendering across
// multiple GPUs.  The only meaningful multi-device workflows are:
//
//   1. Device selection — choose which MTLDevice to use for presentation
//      (relevant on Intel Macs with discrete + integrated GPU; irrelevant on
//      Apple Silicon which has exactly one Metal device).
//
//   2. Compute offload — dispatch heavy, non-rendering compute work
//      (density preparation, RMSD batch computation, volume FFTs) to a
//      secondary MTLDevice while the primary device renders.
//
// The SplitFrame, Alternating, and TaskBased strategies from the original
// PR #186 draft are retained as named constants for API compatibility but
// map to NoOp internally.

#pragma once

#import <Metal/Metal.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace chimerax {
namespace graphics_metal {

// Forward declarations
class MetalContext;
class MetalEventManager;

// ---------------------------------------------------------------------------
// Strategy enumeration
// ---------------------------------------------------------------------------

enum class MultiGPUStrategy {
    // Supported on macOS Metal:
    DeviceSelection  = 0,  // Choose which GPU renders (and presents).
    ComputeOffload   = 1,  // Secondary device handles async compute.

    // NOT supported on macOS Metal — kept for API compatibility only:
    SplitFrame       = 2,  // Stub; maps to DeviceSelection internally.
    TaskBased        = 3,  // Stub; maps to DeviceSelection internally.
    Alternating      = 4,  // Stub; maps to DeviceSelection internally.
};

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

struct GPUDeviceInfo {
    std::string name;
    bool isPrimary;
    bool isActive;
    bool unifiedMemory;
    uint64_t memorySize;    // recommended max working set, bytes
    bool isLowPower;        // integrated / low-power device
};

// ---------------------------------------------------------------------------
// MetalMultiGPU
// ---------------------------------------------------------------------------

class MetalMultiGPU {
public:
    MetalMultiGPU();
    ~MetalMultiGPU();

    bool initialize(MetalContext* context);

    // Device inventory
    std::vector<GPUDeviceInfo> getDeviceInfo() const;

    // Enable / disable and set strategy.
    // Returns false if the strategy requires hardware not present (e.g.
    // SplitFrame always returns false on Apple Silicon).
    bool enable(bool enabled, MultiGPUStrategy strategy = MultiGPUStrategy::DeviceSelection);
    bool isEnabled() const { return _enabled; }
    MultiGPUStrategy getStrategy() const { return _strategy; }

    // Select a named device for presentation (DeviceSelection strategy).
    // Pass an empty string to revert to the system default.
    bool selectPresentationDevice(const std::string& deviceName);
    id<MTLDevice> presentationDevice() const { return _presentationDevice; }

    // Select a named device for async compute offload (ComputeOffload strategy).
    bool selectComputeDevice(const std::string& deviceName);
    id<MTLDevice> computeDevice() const { return _computeDevice; }

    // Submit a compute command buffer to the offload device and return
    // a shared event that signals when work completes.
    id<MTLSharedEvent> submitComputeWork(
        id<MTLCommandBuffer> computeCommandBuffer,
        id<MTLDevice> fromDevice);

private:
    MetalContext*               _context;
    bool                        _enabled;
    MultiGPUStrategy            _strategy;
    std::vector<id<MTLDevice>>  _allDevices;

    id<MTLDevice>               _presentationDevice;
    id<MTLDevice>               _computeDevice;
    id<MTLCommandQueue>         _computeQueue;

    MetalEventManager*          _eventManager;

    // Internal helpers
    id<MTLDevice> _findDeviceByName(const std::string& name) const;
    void          _logUnsupportedStrategy(MultiGPUStrategy strategy) const;
};

} // namespace graphics_metal
} // namespace chimerax
