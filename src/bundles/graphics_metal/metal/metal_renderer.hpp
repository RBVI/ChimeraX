// metal_renderer.hpp
// High-performance Metal renderer for ChimeraX.
//
// Key design decisions for Apple Silicon:
//
// 1. ONE command buffer per frame, ONE render pass encoder for all opaque
//    draw calls, ONE for transparent — avoids per-object command buffer
//    overhead that previously caused multiple presentDrawable calls.
//
// 2. TRIPLE-BUFFERED uniforms + dispatch_semaphore: lets the CPU encode
//    frame N+1 while the GPU is still rasterising frame N.  This keeps
//    all shader cores fed with work at all times.
//
// 3. PERSISTENT geometry buffers keyed by (drawing_id, attribute):
//    allocated once with MTLResourceStorageModeShared (zero-copy on
//    Apple Silicon unified memory), then updated with memcpy when a
//    Drawing marks itself dirty.  No per-frame MTLBuffer allocation.
//
// 4. GPU INSTANCING via [[instance_id]]: drawings with N position copies
//    (symmetry, water molecules, atoms) issue ONE draw call with
//    instanceCount=N.  The per-instance 4×4 transform is stored in a
//    shared-mode buffer updated from Python with the positions array.
//
// 5. DEPTH PRE-PASS for opaque geometry: rasterises depth only, discards
//    fragment colour.  The TBDR (Tile-Based Deferred Rendering) hardware
//    on Apple Silicon can then skip fragment shading for all pixels that
//    fail the depth test — huge win for complex molecular surfaces.
//
// 6. SEPARATE transparent pipeline: depth write off, back-to-front draw
//    order, alpha blending on.  Transparent pass runs AFTER the full
//    opaque pass so all opaque depth is settled.

#pragma once

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <dispatch/dispatch.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace chimerax {
namespace graphics_metal {

// Forward declarations
class MetalContext;
class MetalScene;

// ---------------------------------------------------------------------------
// Uniform block — MSL alignment rules (std140-like but Metal-specific):
//   - float3 pads to float4 (16 bytes)
//   - float4x4 is 64 bytes, 16-byte aligned
// Total = 4*64 + 16 + 16 + 16 + 16 + 16 + 16 = 256 + 96 = 352 bytes.
// Padded to the next 256-byte boundary (256 is the min uniform buffer offset
// alignment on Apple Silicon) → 512 bytes per frame slot.
// ---------------------------------------------------------------------------
struct Uniforms {
    simd::float4x4 modelMatrix;        // 64
    simd::float4x4 viewMatrix;         // 64
    simd::float4x4 projectionMatrix;   // 64
    simd::float4x4 normalMatrix;       // 64  (inverse-transpose of model)

    simd::float3 cameraPosition;
    float         _pad0;               // 16

    simd::float3 lightPosition;
    float        lightRadius;          // 16

    simd::float3 lightColor;
    float        lightIntensity;       // 16

    simd::float3 ambientColor;
    float        ambientIntensity;     // 16

    simd::float2 screenSize;           // for FXAA / silhouette
    float        nearPlane;
    float        farPlane;             // 16
};
static_assert(sizeof(Uniforms) <= 512, "Uniforms must fit in one 512-byte slot");

// Pad each frame's Uniforms to 256 bytes (minimum offset alignment on A-series).
static constexpr size_t kUniformsSlotSize = 512;
static constexpr int    kFrameCount       = 3;   // triple-buffer depth

// ---------------------------------------------------------------------------
// Persistent geometry entry in the buffer pool
// ---------------------------------------------------------------------------
struct GeomBuffer {
    id<MTLBuffer> buffer = nil;  // shared-mode, zero-copy on Apple Silicon
    size_t        capacity = 0;  // bytes currently allocated
};

// Key for the geometry buffer pool: (drawing pointer, attribute tag)
struct GeomKey {
    uintptr_t drawingId;
    uint32_t  attr;   // 0=verts, 1=normals, 2=colors, 3=indices, 4=instances
    bool operator==(const GeomKey& o) const noexcept {
        return drawingId == o.drawingId && attr == o.attr;
    }
};
struct GeomKeyHash {
    size_t operator()(const GeomKey& k) const noexcept {
        return k.drawingId ^ (static_cast<size_t>(k.attr) << 48);
    }
};

// ---------------------------------------------------------------------------
// Pending draw call accumulated during frame encoding
// ---------------------------------------------------------------------------
struct DrawCall {
    id<MTLBuffer> vertexBuf;
    id<MTLBuffer> normalBuf;
    id<MTLBuffer> colorBuf;
    id<MTLBuffer> indexBuf;
    id<MTLBuffer> instanceBuf;   // nil = no instancing
    uint32_t      indexCount;
    uint32_t      instanceCount; // >= 1
    bool          transparent;
    float         sortDepth;     // camera-space z for sort
};

// ---------------------------------------------------------------------------
// MetalRenderer
// ---------------------------------------------------------------------------
class MetalRenderer {
public:
    MetalRenderer(MetalContext* context);
    ~MetalRenderer();

    bool initialize();

    // Scene link
    void setScene(MetalScene* scene) { _scene = scene; }
    MetalScene* scene() const { return _scene; }

    // --- Frame lifecycle ---

    // Call once per frame before any addTriangles calls.
    // Waits on the triple-buffer semaphore, advances frameIndex,
    // acquires drawable and render pass descriptor.
    bool beginFrame(MTKView* view);

    // Call after all addTriangles / addSpheres calls.
    // Executes depth pre-pass, opaque pass, transparent pass,
    // presents the drawable, commits the command buffer, and signals
    // the semaphore in the GPU completion handler.
    void endFrame();

    // --- Geometry accumulation (called by Python drawing_walker) ---

    // Ensure a shared-mode buffer of at least `length` bytes exists for
    // (drawingId, attr) and upload `data` into it.  Returns the buffer.
    // On Apple Silicon "upload" = memcpy into already-GPU-visible memory.
    id<MTLBuffer> ensureBuffer(uintptr_t drawingId, uint32_t attr,
                               const void* data, size_t length);

    // Accumulate one draw call.  Buffers must have been prepared with
    // ensureBuffer().  instanceBuf may be nil for non-instanced draws.
    void addTriangles(id<MTLBuffer> vertexBuf,
                      id<MTLBuffer> normalBuf,
                      id<MTLBuffer> colorBuf,
                      id<MTLBuffer> indexBuf,
                      uint32_t indexCount,
                      id<MTLBuffer> instanceBuf,
                      uint32_t instanceCount,
                      bool transparent,
                      float sortDepth = 0.0f);

    // Convenience: upload raw bytes then accumulate.
    // Called from Cython for drawings whose geometry changed this frame.
    void addTrianglesBytes(uintptr_t drawingId,
                           const void* vertexData,  size_t vertexBytes,
                           const void* normalData,  size_t normalBytes,
                           const void* colorData,   size_t colorBytes,
                           const void* indexData,   size_t indexBytes,
                           uint32_t indexCount,
                           const void* instanceData, size_t instanceBytes,
                           uint32_t instanceCount,
                           bool transparent,
                           float sortDepth = 0.0f);

    // Invalidate cached geometry buffers for a drawing (call when deleted).
    void evictDrawing(uintptr_t drawingId);

    // Multi-device compute offload
    void setComputeOffloadDevice(id<MTLDevice> device);
    void setMultiGPUMode(bool enabled, int strategyInt);

    // Update per-frame scene uniforms (camera, lighting).
    void updateSceneUniforms(const Uniforms& u);

private:
    MetalContext* _context;
    MetalScene*   _scene;

    // --- Pipelines ---
    id<MTLRenderPipelineState> _triangleOpaquePSO;     // depth+color
    id<MTLRenderPipelineState> _triangleTransparentPSO; // blending, no depth write
    id<MTLRenderPipelineState> _depthOnlyPSO;           // pre-pass
    id<MTLDepthStencilState>   _depthWriteState;        // write + test
    id<MTLDepthStencilState>   _depthReadState;         // test only (transparent)
    id<MTLDepthStencilState>   _depthPrePassState;      // write, no colour

    bool createPipelines();

    // --- Triple-buffered uniforms ---
    id<MTLBuffer>       _uniformBuffers[kFrameCount]; // each kUniformsSlotSize bytes
    int                 _frameIndex = 0;
    dispatch_semaphore_t _frameSemaphore;

    Uniforms* _currentFrameUniforms() const {
        return reinterpret_cast<Uniforms*>(
            reinterpret_cast<uint8_t*>([_uniformBuffers[_frameIndex] contents]));
    }
    NSUInteger _uniformOffset() const {
        return static_cast<NSUInteger>(_frameIndex) * kUniformsSlotSize;
    }

    // --- Per-frame encoding state ---
    id<MTLCommandBuffer>        _commandBuffer;
    id<MTLRenderCommandEncoder> _opaqueEncoder;
    id<MTLRenderCommandEncoder> _transparentEncoder;
    id<CAMetalDrawable>         _currentDrawable;
    MTKView*                    _activeView;

    // Pending draw calls accumulated between beginFrame / endFrame
    std::vector<DrawCall> _opaqueCalls;
    std::vector<DrawCall> _transparentCalls;

    // --- Persistent geometry buffer pool ---
    std::unordered_map<GeomKey, GeomBuffer, GeomKeyHash> _geomPool;

    GeomBuffer& _getOrCreate(uintptr_t drawingId, uint32_t attr,
                             size_t minBytes);

    // --- Compute offload ---
    id<MTLDevice>       _computeDevice;
    id<MTLCommandQueue> _computeQueue;
};

} // namespace graphics_metal
} // namespace chimerax
