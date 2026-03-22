// metal_renderer.cpp
// High-performance Metal renderer — see metal_renderer.hpp for design notes.

#include "metal_renderer.hpp"
#include "metal_context.hpp"
#include "metal_resources.hpp"
#include "metal_scene.hpp"
#include <iostream>
#include <algorithm>

namespace chimerax {
namespace graphics_metal {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

MetalRenderer::MetalRenderer(MetalContext* context)
    : _context(context)
    , _scene(nullptr)
    , _triangleOpaquePSO(nil)
    , _triangleTransparentPSO(nil)
    , _depthOnlyPSO(nil)
    , _depthWriteState(nil)
    , _depthReadState(nil)
    , _depthPrePassState(nil)
    , _frameIndex(0)
    , _frameSemaphore(dispatch_semaphore_create(kFrameCount))
    , _commandBuffer(nil)
    , _opaqueEncoder(nil)
    , _transparentEncoder(nil)
    , _currentDrawable(nil)
    , _activeView(nil)
    , _computeDevice(nil)
    , _computeQueue(nil)
{
    for (int i = 0; i < kFrameCount; ++i) _uniformBuffers[i] = nil;
}

MetalRenderer::~MetalRenderer()
{
    for (int i = 0; i < kFrameCount; ++i) {
        if (_uniformBuffers[i]) { [_uniformBuffers[i] release]; }
    }
    dispatch_release(_frameSemaphore);
    for (auto& kv : _geomPool) {
        if (kv.second.buffer) [kv.second.buffer release];
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

bool MetalRenderer::initialize()
{
    if (!_context || !_context->isInitialized()) {
        std::cerr << "MetalRenderer::initialize: no valid Metal context\n";
        return false;
    }

    id<MTLDevice> device = _context->device();

    // Allocate triple-buffered uniform buffers in shared storage mode.
    // On Apple Silicon, shared = unified physical memory, zero DMA copy.
    for (int i = 0; i < kFrameCount; ++i) {
        _uniformBuffers[i] = [device newBufferWithLength:kUniformsSlotSize
                                                 options:MTLResourceStorageModeShared];
        if (!_uniformBuffers[i]) {
            std::cerr << "MetalRenderer: failed to allocate uniform buffer " << i << "\n";
            return false;
        }
        NSString* lbl = [NSString stringWithFormat:@"Uniforms[%d]", i];
        [_uniformBuffers[i] setLabel:lbl];
        // Zero-initialise
        memset([_uniformBuffers[i] contents], 0, kUniformsSlotSize);
    }

    return createPipelines();
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

bool MetalRenderer::createPipelines()
{
    MetalResources* res = _context->resources();
    if (!res->initialize()) return false;

    // Opaque depth state: write + test.
    _depthWriteState = res->createDepthStencilState(true, true,
                                                     MTLCompareFunctionLess);
    // Transparent depth state: test only, no write.
    _depthReadState  = res->createDepthStencilState(true, false,
                                                     MTLCompareFunctionLess);
    // Depth pre-pass state: write + test (same as opaque but colour masked).
    _depthPrePassState = _depthWriteState;

    if (!_depthWriteState || !_depthReadState) return false;

    // Opaque triangle pipeline (no blending).
    _triangleOpaquePSO = res->createRenderPipelineState(
        "vertexTriangle", "fragmentTriangle",
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, false);

    // Transparent triangle pipeline (src-alpha / one-minus-src-alpha blending).
    _triangleTransparentPSO = res->createRenderPipelineState(
        "vertexTriangle", "fragmentTriangle",
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, true);

    // Depth-only pipeline: same vertex shader, null fragment function.
    // Metal requires a fragment function; we use a trivial one that discards.
    _depthOnlyPSO = res->createRenderPipelineState(
        "vertexTriangle", "fragmentDepthOnly",
        MTLPixelFormatInvalid,    // no colour attachment
        MTLPixelFormatDepth32Float, false);

    if (!_triangleOpaquePSO || !_triangleTransparentPSO || !_depthOnlyPSO) {
        std::cerr << "MetalRenderer: pipeline creation failed — "
                     "shaders may not have compiled. "
                     "Run 'make shaders' in src/bundles/graphics_metal/.\n";
        // Non-fatal: fall back to opaque-only rendering without depth pre-pass.
    }

    return true;
}

// ---------------------------------------------------------------------------
// Frame lifecycle
// ---------------------------------------------------------------------------

bool MetalRenderer::beginFrame(MTKView* view)
{
    // Block until fewer than kFrameCount frames are in-flight.
    // This ensures we never overwrite a uniform buffer the GPU is still reading.
    dispatch_semaphore_wait(_frameSemaphore, DISPATCH_TIME_FOREVER);

    _frameIndex = (_frameIndex + 1) % kFrameCount;
    _activeView = view;

    // Grab drawable and render pass early to detect driver/surface issues.
    _currentDrawable       = [view currentDrawable];
    MTLRenderPassDescriptor* rpd = [view currentRenderPassDescriptor];
    if (!_currentDrawable || !rpd) {
        // Drawable not ready; signal semaphore to avoid deadlock.
        dispatch_semaphore_signal(_frameSemaphore);
        return false;
    }

    // Configure render pass:
    //   • loadAction  = Clear (GPU clears the tile at start of render pass,
    //                    cheaper than a separate clear draw call on TBDR)
    //   • storeAction = Store (colour) / DontCare (depth, after depth pre-pass)
    rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0, 0, 0, 1);

    if (rpd.depthAttachment) {
        rpd.depthAttachment.loadAction  = MTLLoadActionClear;
        // After the transparent pass we don't need depth in RAM.
        rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
        rpd.depthAttachment.clearDepth  = 1.0;
    }

    id<MTLCommandQueue> queue = _context->commandQueue();
    _commandBuffer = [queue commandBuffer];
    [_commandBuffer setLabel:@"ChimeraX frame"];

    // Pre-initialise uniform data for this frame slot (Python will overwrite
    // via updateSceneUniforms before any draw calls are encoded).
    Uniforms* u = _currentFrameUniforms();
    u->modelMatrix       = matrix_identity_float4x4;
    u->viewMatrix        = matrix_identity_float4x4;
    u->projectionMatrix  = matrix_identity_float4x4;
    u->normalMatrix      = matrix_identity_float4x4;
    u->cameraPosition    = simd_make_float3(0, 0, 5);
    u->lightPosition     = simd_make_float3(1, 3, 5);
    u->lightRadius       = 100.0f;
    u->lightColor        = simd_make_float3(1, 1, 1);
    u->lightIntensity    = 1.0f;
    u->ambientColor      = simd_make_float3(0.15f, 0.15f, 0.15f);
    u->ambientIntensity  = 1.0f;
    u->screenSize        = simd_make_float2((float)[view drawableSize].width,
                                            (float)[view drawableSize].height);
    u->nearPlane         = 0.1f;
    u->farPlane          = 10000.0f;

    // The actual render pass encoders are opened in endFrame after we have
    // all draw calls collected (so we can depth-sort transparents first).
    _opaqueCalls.clear();
    _transparentCalls.clear();

    return true;
}

void MetalRenderer::endFrame()
{
    if (!_commandBuffer || !_currentDrawable) return;

    MTKView* view = _activeView;
    MTLRenderPassDescriptor* rpd = [view currentRenderPassDescriptor];
    if (!rpd) { [_commandBuffer commit]; return; }

    // --- Depth pre-pass (opaque geometry only) ---
    // On TBDR hardware this lets the GPU eliminate occluded fragments before
    // the main fragment shading pass.  Worth ~30-60% fragment work reduction
    // for dense molecular surfaces with many overlapping layers.
    if (_depthOnlyPSO && !_opaqueCalls.empty()) {
        // Depth-only render pass — disable colour attachment.
        MTLRenderPassDescriptor* depthRPD = [MTLRenderPassDescriptor renderPassDescriptor];
        depthRPD.depthAttachment.texture    = rpd.depthAttachment.texture;
        depthRPD.depthAttachment.loadAction = MTLLoadActionClear;
        depthRPD.depthAttachment.storeAction = MTLStoreActionStore; // keep for main pass
        depthRPD.depthAttachment.clearDepth = 1.0;

        id<MTLRenderCommandEncoder> depthEnc =
            [_commandBuffer renderCommandEncoderWithDescriptor:depthRPD];
        [depthEnc setLabel:@"DepthPrePass"];
        [depthEnc setRenderPipelineState:_depthOnlyPSO];
        [depthEnc setDepthStencilState:_depthPrePassState];
        [depthEnc setCullMode:MTLCullModeBack];

        NSUInteger uOff = _uniformOffset();
        for (const DrawCall& dc : _opaqueCalls) {
            [depthEnc setVertexBuffer:dc.vertexBuf offset:0 atIndex:0];
            if (dc.instanceBuf) {
                [depthEnc setVertexBuffer:dc.instanceBuf offset:0 atIndex:4];
            }
            [depthEnc setVertexBuffer:_uniformBuffers[_frameIndex] offset:uOff atIndex:3];

            if (dc.instanceBuf) {
                [depthEnc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                     indexCount:dc.indexCount
                                      indexType:MTLIndexTypeUInt32
                                    indexBuffer:dc.indexBuf
                              indexBufferOffset:0
                                  instanceCount:dc.instanceCount
                                     baseVertex:0
                                   baseInstance:0];
            } else {
                [depthEnc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                     indexCount:dc.indexCount
                                      indexType:MTLIndexTypeUInt32
                                    indexBuffer:dc.indexBuf
                              indexBufferOffset:0];
            }
        }
        [depthEnc endEncoding];

        // Change depth attachment for main pass to load the pre-pass result.
        rpd.depthAttachment.loadAction = MTLLoadActionLoad;
    }

    NSUInteger uOff = _uniformOffset();

    // --- Main opaque pass ---
    if (!_opaqueCalls.empty() && _triangleOpaquePSO) {
        id<MTLRenderCommandEncoder> enc =
            [_commandBuffer renderCommandEncoderWithDescriptor:rpd];
        [enc setLabel:@"OpaquePass"];
        [enc setRenderPipelineState:_triangleOpaquePSO];
        [enc setDepthStencilState:_depthWriteState];
        [enc setCullMode:MTLCullModeBack];

        for (const DrawCall& dc : _opaqueCalls) {
            [enc setVertexBuffer:dc.vertexBuf   offset:0    atIndex:0];
            [enc setVertexBuffer:dc.colorBuf    offset:0    atIndex:1];
            [enc setVertexBuffer:dc.normalBuf   offset:0    atIndex:2];
            [enc setVertexBuffer:_uniformBuffers[_frameIndex] offset:uOff atIndex:3];
            if (dc.instanceBuf)
                [enc setVertexBuffer:dc.instanceBuf offset:0 atIndex:4];
            [enc setFragmentBuffer:_uniformBuffers[_frameIndex] offset:uOff atIndex:0];

            if (dc.instanceBuf) {
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:dc.indexCount
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:dc.indexBuf
                         indexBufferOffset:0
                             instanceCount:dc.instanceCount
                                baseVertex:0
                              baseInstance:0];
            } else {
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:dc.indexCount
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:dc.indexBuf
                         indexBufferOffset:0];
            }
        }
        [enc endEncoding];

        // Depth pre-pass already wrote depth; subsequent passes just test.
        rpd.depthAttachment.loadAction = MTLLoadActionLoad;
        rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
    }

    // --- Transparent pass (back-to-front, depth test only) ---
    if (!_transparentCalls.empty() && _triangleTransparentPSO) {
        // Sort back-to-front.
        std::sort(_transparentCalls.begin(), _transparentCalls.end(),
                  [](const DrawCall& a, const DrawCall& b){
                      return a.sortDepth < b.sortDepth;  // larger z = further = first
                  });

        id<MTLRenderCommandEncoder> enc =
            [_commandBuffer renderCommandEncoderWithDescriptor:rpd];
        [enc setLabel:@"TransparentPass"];
        [enc setRenderPipelineState:_triangleTransparentPSO];
        [enc setDepthStencilState:_depthReadState];
        [enc setCullMode:MTLCullModeNone];  // show both faces for transparent geometry

        for (const DrawCall& dc : _transparentCalls) {
            [enc setVertexBuffer:dc.vertexBuf   offset:0    atIndex:0];
            [enc setVertexBuffer:dc.colorBuf    offset:0    atIndex:1];
            [enc setVertexBuffer:dc.normalBuf   offset:0    atIndex:2];
            [enc setVertexBuffer:_uniformBuffers[_frameIndex] offset:uOff atIndex:3];
            if (dc.instanceBuf)
                [enc setVertexBuffer:dc.instanceBuf offset:0 atIndex:4];
            [enc setFragmentBuffer:_uniformBuffers[_frameIndex] offset:uOff atIndex:0];

            if (dc.instanceBuf) {
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:dc.indexCount
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:dc.indexBuf
                         indexBufferOffset:0
                             instanceCount:dc.instanceCount
                                baseVertex:0
                              baseInstance:0];
            } else {
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:dc.indexCount
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:dc.indexBuf
                         indexBufferOffset:0];
            }
        }
        [enc endEncoding];
    }

    // Present and commit.
    [_commandBuffer presentDrawable:_currentDrawable];

    // Signal the semaphore in the completion handler so the CPU can begin
    // encoding the next frame as soon as the GPU has freed this frame's slot.
    dispatch_semaphore_t sem = _frameSemaphore;
    [_commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        dispatch_semaphore_signal(sem);
    }];
    [_commandBuffer commit];

    _commandBuffer   = nil;
    _currentDrawable = nil;
    _activeView      = nil;
}

// ---------------------------------------------------------------------------
// Scene uniform update (called from Python MetalBackend._sync_camera)
// ---------------------------------------------------------------------------

void MetalRenderer::updateSceneUniforms(const Uniforms& u)
{
    Uniforms* dst = _currentFrameUniforms();
    if (dst) *dst = u;
}

// ---------------------------------------------------------------------------
// Persistent buffer pool
// ---------------------------------------------------------------------------

GeomBuffer& MetalRenderer::_getOrCreate(uintptr_t drawingId, uint32_t attr,
                                         size_t minBytes)
{
    GeomKey key{drawingId, attr};
    auto it = _geomPool.find(key);
    if (it != _geomPool.end()) {
        if (it->second.capacity >= minBytes) return it->second;
        // Need to grow — release old.
        if (it->second.buffer) [it->second.buffer release];
        _geomPool.erase(it);
    }

    // Round up to 4 KB alignment for better page utilisation.
    size_t sz = (minBytes + 4095) & ~4095ULL;

    id<MTLBuffer> buf = [_context->device()
        newBufferWithLength:sz
                    options:MTLResourceStorageModeShared];
    if (!buf) {
        // Fallback: return a zero-capacity entry that callers must check.
        _geomPool[key] = GeomBuffer{nil, 0};
        return _geomPool[key];
    }

    GeomBuffer entry;
    entry.buffer   = buf;
    entry.capacity = sz;
    _geomPool[key] = entry;
    return _geomPool[key];
}

id<MTLBuffer> MetalRenderer::ensureBuffer(uintptr_t drawingId, uint32_t attr,
                                           const void* data, size_t length)
{
    if (length == 0) return nil;
    GeomBuffer& gb = _getOrCreate(drawingId, attr, length);
    if (!gb.buffer) return nil;
    // Zero-copy on Apple Silicon: this memcpy writes into unified DRAM pages
    // that the GPU already has access to via the same MTLBuffer.
    memcpy([gb.buffer contents], data, length);
    return gb.buffer;
}

void MetalRenderer::evictDrawing(uintptr_t drawingId)
{
    for (uint32_t attr = 0; attr <= 4; ++attr) {
        GeomKey key{drawingId, attr};
        auto it = _geomPool.find(key);
        if (it != _geomPool.end()) {
            if (it->second.buffer) [it->second.buffer release];
            _geomPool.erase(it);
        }
    }
}

// ---------------------------------------------------------------------------
// Draw call accumulation
// ---------------------------------------------------------------------------

void MetalRenderer::addTriangles(id<MTLBuffer> vertexBuf,
                                  id<MTLBuffer> normalBuf,
                                  id<MTLBuffer> colorBuf,
                                  id<MTLBuffer> indexBuf,
                                  uint32_t indexCount,
                                  id<MTLBuffer> instanceBuf,
                                  uint32_t instanceCount,
                                  bool transparent,
                                  float sortDepth)
{
    if (!vertexBuf || !indexBuf || indexCount == 0) return;
    DrawCall dc;
    dc.vertexBuf    = vertexBuf;
    dc.normalBuf    = normalBuf;
    dc.colorBuf     = colorBuf;
    dc.indexBuf     = indexBuf;
    dc.instanceBuf  = instanceBuf;
    dc.indexCount   = indexCount;
    dc.instanceCount = std::max(1u, instanceCount);
    dc.transparent  = transparent;
    dc.sortDepth    = sortDepth;

    if (transparent)
        _transparentCalls.push_back(dc);
    else
        _opaqueCalls.push_back(dc);
}

void MetalRenderer::addTrianglesBytes(uintptr_t drawingId,
                                       const void* vertexData,  size_t vertexBytes,
                                       const void* normalData,  size_t normalBytes,
                                       const void* colorData,   size_t colorBytes,
                                       const void* indexData,   size_t indexBytes,
                                       uint32_t indexCount,
                                       const void* instanceData, size_t instanceBytes,
                                       uint32_t instanceCount,
                                       bool transparent,
                                       float sortDepth)
{
    if (!_context || !_context->isInitialized()) return;

    id<MTLBuffer> vbuf = ensureBuffer(drawingId, 0, vertexData,   vertexBytes);
    id<MTLBuffer> nbuf = ensureBuffer(drawingId, 1, normalData,   normalBytes);
    id<MTLBuffer> cbuf = ensureBuffer(drawingId, 2, colorData,    colorBytes);
    id<MTLBuffer> ibuf = ensureBuffer(drawingId, 3, indexData,    indexBytes);
    id<MTLBuffer> instbuf = nil;
    if (instanceData && instanceBytes > 0)
        instbuf = ensureBuffer(drawingId, 4, instanceData, instanceBytes);

    addTriangles(vbuf, nbuf, cbuf, ibuf, indexCount,
                 instbuf, instanceCount, transparent, sortDepth);
}

// ---------------------------------------------------------------------------
// Multi-device compute offload
// ---------------------------------------------------------------------------

void MetalRenderer::setComputeOffloadDevice(id<MTLDevice> device)
{
    _computeDevice = device;
    _computeQueue  = device ? [device newCommandQueue] : nil;
}

void MetalRenderer::setMultiGPUMode(bool enabled, int strategyInt)
{
    if (!enabled) {
        _computeDevice = nil;
        _computeQueue  = nil;
    }
    (void)strategyInt;
}

} // namespace graphics_metal
} // namespace chimerax
