/*
 * Metal context management: device, command queue, CAMetalLayer setup.
 *
 * This is the foundation of the Metal backend. It takes an NSView pointer
 * from Qt's winId() and sets up a CAMetalLayer for rendering.
 */

#include <Python.h>
#include "metal_internal.h"

static MetalContext* get_context(PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    static const char* kwlist[] = {"context", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("l"), (char**)kwlist, &ctx_ptr))
        return nullptr;
    return (MetalContext*)ctx_ptr;
}

// Create depth texture matching drawable size
static void create_depth_texture(MetalContext* ctx) {
    if (ctx->depthTexture &&
        ctx->depthTexture.width == ctx->width &&
        ctx->depthTexture.height == ctx->height)
        return;

    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:ctx->depthPixelFormat
        width:ctx->width
        height:ctx->height
        mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    ctx->depthTexture = [ctx->device newTextureWithDescriptor:desc];
}

extern "C"
PyObject* metal_create_context(PyObject* self, PyObject* args, PyObject* kwds) {
    long nsview_pointer;
    static const char* kwlist[] = {"nsview_pointer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("l"), (char**)kwlist, &nsview_pointer))
        return NULL;

    NSView* view = (NSView*)nsview_pointer;

    // Create Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        PyErr_SetString(PyExc_RuntimeError, "No Metal-capable GPU found");
        return NULL;
    }

    // Create command queue
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Set up CAMetalLayer on the view
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = NO;  // Allow texture reads for post-processing
    layer.contentsScale = view.window.backingScaleFactor;

    // Replace the view's layer
    view.wantsLayer = YES;
    view.layer = layer;

    // Size the layer
    CGSize size = view.bounds.size;
    CGFloat scale = layer.contentsScale;
    layer.drawableSize = CGSizeMake(size.width * scale, size.height * scale);

    // Allocate context
    MetalContext* ctx = new MetalContext();
    ctx->device = device;
    ctx->commandQueue = queue;
    ctx->layer = layer;
    ctx->shaderLibrary = nil;
    ctx->currentDrawable = nil;
    ctx->currentCommandBuffer = nil;
    ctx->currentEncoder = nil;
    ctx->depthTexture = nil;
    ctx->depthPixelFormat = MTLPixelFormatDepth32Float;
    ctx->width = (NSUInteger)(size.width * scale);
    ctx->height = (NSUInteger)(size.height * scale);

    // Create initial depth texture
    create_depth_texture(ctx);

    return PyLong_FromLong((long)ctx);
}

extern "C"
PyObject* metal_destroy_context(PyObject* self, PyObject* args, PyObject* kwds) {
    MetalContext* ctx = get_context(args, kwds);
    if (!ctx) return NULL;

    // Wait for GPU to finish
    if (ctx->currentCommandBuffer) {
        [ctx->currentCommandBuffer waitUntilCompleted];
    }

    ctx->device = nil;
    ctx->commandQueue = nil;
    ctx->layer = nil;
    ctx->shaderLibrary = nil;
    ctx->depthTexture = nil;
    ctx->currentDrawable = nil;
    ctx->currentCommandBuffer = nil;
    ctx->currentEncoder = nil;

    delete ctx;
    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_resize_drawable(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    int width, height;
    static const char* kwlist[] = {"context", "width", "height", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lii"), (char**)kwlist, &ctx_ptr, &width, &height))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    ctx->width = width;
    ctx->height = height;
    ctx->layer.drawableSize = CGSizeMake(width, height);
    create_depth_texture(ctx);

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_begin_frame(PyObject* self, PyObject* args, PyObject* kwds) {
    MetalContext* ctx = get_context(args, kwds);
    if (!ctx) return NULL;

    // Get next drawable
    ctx->currentDrawable = [ctx->layer nextDrawable];
    if (!ctx->currentDrawable) {
        PyErr_SetString(PyExc_RuntimeError, "Could not get next Metal drawable");
        return NULL;
    }

    // Create command buffer
    ctx->currentCommandBuffer = [ctx->commandQueue commandBuffer];

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_end_frame(PyObject* self, PyObject* args, PyObject* kwds) {
    MetalContext* ctx = get_context(args, kwds);
    if (!ctx) return NULL;

    // End any active render pass
    if (ctx->currentEncoder) {
        [ctx->currentEncoder endEncoding];
        ctx->currentEncoder = nil;
    }

    // Present and commit
    if (ctx->currentDrawable && ctx->currentCommandBuffer) {
        [ctx->currentCommandBuffer presentDrawable:ctx->currentDrawable];
        [ctx->currentCommandBuffer commit];
    }

    ctx->currentDrawable = nil;
    ctx->currentCommandBuffer = nil;

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_begin_render_pass(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    float clear_r, clear_g, clear_b, clear_a;
    int clear_depth_flag;
    int load_color = 0;   // 1 = load existing color (no clear)
    int load_depth = 0;   // 1 = load existing depth (no clear)
    static const char* kwlist[] = {"context", "clear_r", "clear_g", "clear_b", "clear_a",
                                   "clear_depth", "load_color", "load_depth", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lffffi|ii"), (char**)kwlist,
            &ctx_ptr, &clear_r, &clear_g, &clear_b, &clear_a, &clear_depth_flag,
            &load_color, &load_depth))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;

    // End previous encoder if active
    if (ctx->currentEncoder) {
        [ctx->currentEncoder endEncoding];
        ctx->currentEncoder = nil;
    }

    // Create render pass descriptor
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];

    // Color attachment — the drawable texture
    rpd.colorAttachments[0].texture = ctx->currentDrawable.texture;
    if (load_color) {
        rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
    } else {
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(clear_r, clear_g, clear_b, clear_a);
    }
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    // Depth attachment
    rpd.depthAttachment.texture = ctx->depthTexture;
    if (load_depth) {
        rpd.depthAttachment.loadAction = MTLLoadActionLoad;
    } else {
        rpd.depthAttachment.loadAction = clear_depth_flag ? MTLLoadActionClear : MTLLoadActionLoad;
        rpd.depthAttachment.clearDepth = 1.0;
    }
    rpd.depthAttachment.storeAction = MTLStoreActionStore;

    // Create encoder
    ctx->currentEncoder = [ctx->currentCommandBuffer renderCommandEncoderWithDescriptor:rpd];
    if (!ctx->currentEncoder) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create render command encoder");
        return NULL;
    }

    // Default state
    [ctx->currentEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
    [ctx->currentEncoder setCullMode:MTLCullModeBack];

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_begin_depth_pass(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, tex_ptr;
    int width, height;
    static const char* kwlist[] = {"context", "depth_texture", "width", "height", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llii"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &width, &height))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    MetalTexture* tex = (MetalTexture*)tex_ptr;

    // End any active encoder
    if (ctx->currentEncoder) {
        [ctx->currentEncoder endEncoding];
        ctx->currentEncoder = nil;
    }

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];

    // No color attachment — depth only
    rpd.colorAttachments[0].texture = nil;
    rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rpd.colorAttachments[0].storeAction = MTLStoreActionDontCare;

    // Depth attachment — the shadow map texture
    rpd.depthAttachment.texture = tex->texture;
    rpd.depthAttachment.loadAction = MTLLoadActionClear;
    rpd.depthAttachment.storeAction = MTLStoreActionStore;
    rpd.depthAttachment.clearDepth = 1.0;

    ctx->currentEncoder = [ctx->currentCommandBuffer renderCommandEncoderWithDescriptor:rpd];
    if (!ctx->currentEncoder) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create depth render command encoder");
        return NULL;
    }

    [ctx->currentEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
    [ctx->currentEncoder setCullMode:MTLCullModeNone];

    MTLViewport vp = {0, 0, (double)width, (double)height, 0.0, 1.0};
    [ctx->currentEncoder setViewport:vp];

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_end_render_pass(PyObject* self, PyObject* args, PyObject* kwds) {
    MetalContext* ctx = get_context(args, kwds);
    if (!ctx) return NULL;

    if (ctx->currentEncoder) {
        [ctx->currentEncoder endEncoding];
        ctx->currentEncoder = nil;
    }

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_set_viewport(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    double x, y, w, h;
    static const char* kwlist[] = {"context", "x", "y", "width", "height", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ldddd"), (char**)kwlist,
            &ctx_ptr, &x, &y, &w, &h))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->currentEncoder) {
        MTLViewport vp = {x, y, w, h, 0.0, 1.0};
        [ctx->currentEncoder setViewport:vp];
    }

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_set_cull_mode(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    int mode;  // 0=none, 1=front, 2=back
    static const char* kwlist[] = {"context", "mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("li"), (char**)kwlist, &ctx_ptr, &mode))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    if (ctx->currentEncoder) {
        MTLCullMode cm = (mode == 0) ? MTLCullModeNone :
                         (mode == 1) ? MTLCullModeFront : MTLCullModeBack;
        [ctx->currentEncoder setCullMode:cm];
    }

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_get_depth_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    MetalContext* ctx = get_context(args, kwds);
    if (!ctx) return NULL;

    if (!ctx->depthTexture) {
        Py_RETURN_NONE;
    }

    // Wrap the depth texture in a MetalTexture struct so it can be used
    // with set_fragment_texture like any other texture.
    MetalTexture* tex = new MetalTexture();
    tex->texture = ctx->depthTexture;
    // Create a sampler with nearest filtering and no depth compare
    // (we want raw depth values, not shadow comparison results).
    MTLSamplerDescriptor* sampDesc = [[MTLSamplerDescriptor alloc] init];
    sampDesc.minFilter = MTLSamplerMinMagFilterNearest;
    sampDesc.magFilter = MTLSamplerMinMagFilterNearest;
    sampDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    tex->sampler = [ctx->device newSamplerStateWithDescriptor:sampDesc];
    tex->pixelFormat = ctx->depthPixelFormat;
    tex->width = ctx->width;
    tex->height = ctx->height;

    return PyLong_FromLong((long)tex);
}

extern "C"
PyObject* metal_begin_color_pass(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, tex_ptr;
    int width, height;
    float clear_r = 0, clear_g = 0, clear_b = 0, clear_a = 0;
    int do_clear = 1;
    static const char* kwlist[] = {"context", "color_texture", "width", "height",
                                   "clear_r", "clear_g", "clear_b", "clear_a",
                                   "do_clear", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llii|ffffi"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &width, &height,
            &clear_r, &clear_g, &clear_b, &clear_a, &do_clear))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    MetalTexture* tex = (MetalTexture*)tex_ptr;

    // End any active encoder
    if (ctx->currentEncoder) {
        [ctx->currentEncoder endEncoding];
        ctx->currentEncoder = nil;
    }

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];

    // Color attachment — the target texture
    rpd.colorAttachments[0].texture = tex->texture;
    if (do_clear) {
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(clear_r, clear_g, clear_b, clear_a);
    } else {
        rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
    }
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    // No depth attachment for post-processing passes
    rpd.depthAttachment.texture = nil;

    ctx->currentEncoder = [ctx->currentCommandBuffer renderCommandEncoderWithDescriptor:rpd];
    if (!ctx->currentEncoder) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create color render command encoder");
        return NULL;
    }

    [ctx->currentEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
    [ctx->currentEncoder setCullMode:MTLCullModeNone];

    MTLViewport vp = {0, 0, (double)width, (double)height, 0.0, 1.0};
    [ctx->currentEncoder setViewport:vp];

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
//  Multi-window: switch the context's CAMetalLayer to a different NSView.
//  Returns the previous layer pointer so it can be restored.
// ---------------------------------------------------------------------------
extern "C"
PyObject* metal_set_layer_for_view(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, nsview_ptr;
    static const char* kwlist[] = {"context", "nsview_pointer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ll"), (char**)kwlist,
            &ctx_ptr, &nsview_ptr))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    NSView* view = (NSView*)nsview_ptr;

    // Save previous layer.
    long prev_layer = (long)(__bridge void *)ctx->layer;

    // Check if this view already has a CAMetalLayer we set up.
    CAMetalLayer* target_layer = nil;
    if ([view.layer isKindOfClass:[CAMetalLayer class]]) {
        target_layer = (CAMetalLayer*)view.layer;
        // Make sure it uses our device.
        if (target_layer.device != ctx->device) {
            target_layer.device = ctx->device;
        }
    } else {
        // Create a new CAMetalLayer for this view.
        target_layer = [CAMetalLayer layer];
        target_layer.device = ctx->device;
        target_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        target_layer.framebufferOnly = NO;
        if (view.window) {
            target_layer.contentsScale = view.window.backingScaleFactor;
        }
        view.wantsLayer = YES;
        view.layer = target_layer;
    }

    // Size the layer.
    CGSize size = view.bounds.size;
    CGFloat scale = target_layer.contentsScale;
    NSUInteger w = (NSUInteger)(size.width * scale);
    NSUInteger h = (NSUInteger)(size.height * scale);
    if (w < 1) w = 1;
    if (h < 1) h = 1;

    // Only resize if dimensions actually changed — setting drawableSize
    // discards the current drawable, causing a visible flash.
    CGSize cur = target_layer.drawableSize;
    if ((NSUInteger)cur.width != w || (NSUInteger)cur.height != h) {
        target_layer.drawableSize = CGSizeMake(w, h);
    }

    // Switch context to the new layer.
    ctx->layer = target_layer;
    ctx->width = w;
    ctx->height = h;

    // Recreate depth texture for new size.
    create_depth_texture(ctx);

    return PyLong_FromLong(prev_layer);
}

extern "C"
PyObject* metal_device_name(PyObject* self, PyObject* args, PyObject* kwds) {
    MetalContext* ctx = get_context(args, kwds);
    if (!ctx) return NULL;

    NSString* name = ctx->device.name;
    return PyUnicode_FromString([name UTF8String]);
}
