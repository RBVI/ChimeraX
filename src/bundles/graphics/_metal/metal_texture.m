/*
 * Metal texture management.
 */

#include <Python.h>
#include "metal_internal.h"

extern "C"
PyObject* metal_create_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    int width, height;
    int format;  // 0=RGBA8, 1=R8, 2=Depth32Float, 3=Depth32Float_Stencil8, 4=R32Float
    int linear_filter;
    int depth_compare;

    static const char* kwlist[] = {"context", "width", "height", "format",
                                   "linear_filter", "depth_compare", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("liiiii"), (char**)kwlist,
            &ctx_ptr, &width, &height, &format, &linear_filter, &depth_compare))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLDevice> device = ctx->device;

    // Map format enum to Metal pixel format
    MTLPixelFormat pixelFormat;
    switch (format) {
        case 0: pixelFormat = MTLPixelFormatRGBA8Unorm; break;
        case 1: pixelFormat = MTLPixelFormatR8Unorm; break;
        case 2: pixelFormat = MTLPixelFormatDepth32Float; break;
        case 3: pixelFormat = MTLPixelFormatDepth32Float_Stencil8; break;
        case 4: pixelFormat = MTLPixelFormatR32Float; break;
        default: pixelFormat = MTLPixelFormatRGBA8Unorm; break;
    }

    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:pixelFormat
        width:width
        height:height
        mipmapped:NO];
    if (format == 2 || format == 3) {
        // Depth textures: read + render target only (no compute write).
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    } else {
        // Color textures (R8, RGBA8, R32Float, etc.): allow compute writes.
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite;
    }
    desc.storageMode = MTLStorageModePrivate;

    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
    if (!texture) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Metal texture");
        return NULL;
    }

    // Create sampler
    MTLSamplerDescriptor* sampDesc = [[MTLSamplerDescriptor alloc] init];
    sampDesc.minFilter = linear_filter ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sampDesc.magFilter = linear_filter ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sampDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;

    if (depth_compare) {
        sampDesc.compareFunction = MTLCompareFunctionLessEqual;
    }

    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:sampDesc];

    MetalTexture* tex = new MetalTexture();
    tex->texture = texture;
    tex->sampler = sampler;
    tex->pixelFormat = pixelFormat;
    tex->width = width;
    tex->height = height;

    return PyLong_FromLong((long)tex);
}

extern "C"
PyObject* metal_destroy_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    long tex_ptr;
    static const char* kwlist[] = {"texture", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("l"), (char**)kwlist, &tex_ptr))
        return NULL;

    MetalTexture* tex = (MetalTexture*)tex_ptr;
    tex->texture = nil;
    tex->sampler = nil;
    delete tex;

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_set_fragment_texture(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, tex_ptr;
    int index;
    static const char* kwlist[] = {"context", "texture", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lli"), (char**)kwlist,
            &ctx_ptr, &tex_ptr, &index))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    MetalTexture* tex = (MetalTexture*)tex_ptr;

    // Get encoder from context
    id<MTLRenderCommandEncoder> encoder = ctx->currentEncoder;

    if (encoder) {
        [encoder setFragmentTexture:tex->texture atIndex:index];
        [encoder setFragmentSamplerState:tex->sampler atIndex:index];
    }

    Py_RETURN_NONE;
}
