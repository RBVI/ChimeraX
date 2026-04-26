/*
 * Metal pipeline state management: shader compilation, pipeline cache.
 *
 * Compiles .metal shader files into MTLLibrary, then creates
 * MTLRenderPipelineState objects with function constants for
 * capability-based feature toggling.
 */

#include <Python.h>
#include "metal_internal.h"

extern "C"
PyObject* metal_create_pipeline(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    const char* shader_source;
    const char* vertex_func_name;
    const char* fragment_func_name;
    int has_depth;
    int depth_only = 0;
    int color_write = 1;
    // Function constant values as a flat array of ints (0 or 1)
    PyObject* constants_list;

    static const char* kwlist[] = {"context", "shader_source", "vertex_function",
                                   "fragment_function", "has_depth", "constants",
                                   "depth_only", "color_write", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lsssiO|ii"), (char**)kwlist,
            &ctx_ptr, &shader_source, &vertex_func_name, &fragment_func_name,
            &has_depth, &constants_list, &depth_only, &color_write))
        return NULL;

    // Get device from context
    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLDevice> device = ctx->device;

    // Compile shader source
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:shader_source];

    // Build function constant values from Python list
    MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
    if (PyList_Check(constants_list)) {
        Py_ssize_t n = PyList_Size(constants_list);
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject* item = PyList_GetItem(constants_list, i);
            bool val = (PyLong_AsLong(item) != 0);
            [fcv setConstantValue:&val type:MTLDataTypeBool atIndex:i];
        }
    }

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        NSString* errStr = [error localizedDescription];
        PyErr_Format(PyExc_RuntimeError, "Metal shader compilation failed: %s",
                     [errStr UTF8String]);
        return NULL;
    }

    // Get functions with constants
    NSString* vName = [NSString stringWithUTF8String:vertex_func_name];
    NSString* fName = [NSString stringWithUTF8String:fragment_func_name];

    id<MTLFunction> vertexFunc = [library newFunctionWithName:vName
                                         constantValues:fcv error:&error];
    if (!vertexFunc) {
        // Try without constants (post-processing shaders may not use them)
        vertexFunc = [library newFunctionWithName:vName];
    }
    if (!vertexFunc) {
        PyErr_Format(PyExc_RuntimeError, "Vertex function '%s' not found", vertex_func_name);
        return NULL;
    }

    id<MTLFunction> fragmentFunc = [library newFunctionWithName:fName
                                           constantValues:fcv error:&error];
    if (!fragmentFunc) {
        fragmentFunc = [library newFunctionWithName:fName];
    }
    if (!fragmentFunc) {
        PyErr_Format(PyExc_RuntimeError, "Fragment function '%s' not found", fragment_func_name);
        return NULL;
    }

    // Create pipeline descriptor
    MTLRenderPipelineDescriptor* pipeDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipeDesc.vertexFunction = vertexFunc;
    pipeDesc.fragmentFunction = fragmentFunc;
    if (depth_only) {
        pipeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatInvalid;
    } else {
        pipeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        pipeDesc.colorAttachments[0].writeMask = color_write ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
        pipeDesc.colorAttachments[0].blendingEnabled = YES;
        pipeDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pipeDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipeDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pipeDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    }

    if (has_depth) {
        pipeDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    }

    // Vertex descriptor for standard attribute layout.
    // Buffer indices 0-3 are reserved for uniform/constant buffers.
    // Vertex data starts at buffer index 10.
    MTLVertexDescriptor* vertDesc = [[MTLVertexDescriptor alloc] init];

    // Attribute 0: position (float3) — buffer 10
    vertDesc.attributes[0].format = MTLVertexFormatFloat3;
    vertDesc.attributes[0].offset = 0;
    vertDesc.attributes[0].bufferIndex = 10;
    vertDesc.layouts[10].stride = 12;
    vertDesc.layouts[10].stepFunction = MTLVertexStepFunctionPerVertex;

    // Attribute 1: tex_coord (float2) — buffer 11
    vertDesc.attributes[1].format = MTLVertexFormatFloat2;
    vertDesc.attributes[1].offset = 0;
    vertDesc.attributes[1].bufferIndex = 11;
    vertDesc.layouts[11].stride = 8;
    vertDesc.layouts[11].stepFunction = MTLVertexStepFunctionPerVertex;

    // Attribute 2: normal (float3) — buffer 12
    vertDesc.attributes[2].format = MTLVertexFormatFloat3;
    vertDesc.attributes[2].offset = 0;
    vertDesc.attributes[2].bufferIndex = 12;
    vertDesc.layouts[12].stride = 12;
    vertDesc.layouts[12].stepFunction = MTLVertexStepFunctionPerVertex;

    // Attribute 3: vcolor (uchar4 normalized) — buffer 13, per instance
    vertDesc.attributes[3].format = MTLVertexFormatUChar4Normalized;
    vertDesc.attributes[3].offset = 0;
    vertDesc.attributes[3].bufferIndex = 13;
    vertDesc.layouts[13].stride = 4;
    vertDesc.layouts[13].stepFunction = MTLVertexStepFunctionPerInstance;

    // Attribute 4: instance_shift_and_scale (float4) — buffer 14, per instance
    vertDesc.attributes[4].format = MTLVertexFormatFloat4;
    vertDesc.attributes[4].offset = 0;
    vertDesc.attributes[4].bufferIndex = 14;
    vertDesc.layouts[14].stride = 16;
    vertDesc.layouts[14].stepFunction = MTLVertexStepFunctionPerInstance;

    // Attributes 5-8: instance_placement (float4x4) — buffer 15, per instance
    for (int i = 0; i < 4; i++) {
        vertDesc.attributes[5 + i].format = MTLVertexFormatFloat4;
        vertDesc.attributes[5 + i].offset = i * 16;
        vertDesc.attributes[5 + i].bufferIndex = 15;
    }
    vertDesc.layouts[15].stride = 64;
    vertDesc.layouts[15].stepFunction = MTLVertexStepFunctionPerInstance;

    pipeDesc.vertexDescriptor = vertDesc;

    // Create pipeline state
    id<MTLRenderPipelineState> pipeState = [device newRenderPipelineStateWithDescriptor:pipeDesc
                                                    error:&error];
    if (!pipeState) {
        NSString* errStr = [error localizedDescription];
        PyErr_Format(PyExc_RuntimeError, "Pipeline creation failed: %s", [errStr UTF8String]);
        return NULL;
    }

    // Create depth stencil state
    id<MTLDepthStencilState> dsState = nil;
    if (has_depth) {
        MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
        dsDesc.depthCompareFunction = MTLCompareFunctionLess;
        dsDesc.depthWriteEnabled = YES;
        dsState = [device newDepthStencilStateWithDescriptor:dsDesc];
    }

    MetalPipeline* pipeline = new MetalPipeline();
    pipeline->state = pipeState;
    pipeline->depthStencilState = dsState;

    return PyLong_FromLong((long)pipeline);
}

extern "C"
PyObject* metal_set_pipeline(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, pipeline_ptr;
    static const char* kwlist[] = {"context", "pipeline", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ll"), (char**)kwlist, &ctx_ptr, &pipeline_ptr))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    MetalPipeline* pipeline = (MetalPipeline*)pipeline_ptr;

    // Access current encoder through context (third field after device, queue)
    id<MTLRenderCommandEncoder> encoder = ctx->currentEncoder;

    if (encoder) {
        [encoder setRenderPipelineState:pipeline->state];
        if (pipeline->depthStencilState) {
            [encoder setDepthStencilState:pipeline->depthStencilState];
        }
    }

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_set_depth_stencil_state(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    int depth_write, depth_test;
    int compare_func;  // 0=never, 1=less, 2=equal, 3=lequal, 4=greater, 5=notequal, 6=gequal, 7=always
    static const char* kwlist[] = {"context", "depth_write", "depth_test", "compare_func", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("liii"), (char**)kwlist,
            &ctx_ptr, &depth_write, &depth_test, &compare_func))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    id<MTLDevice> device = ctx->device;

    MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
    desc.depthWriteEnabled = depth_write ? YES : NO;

    MTLCompareFunction funcs[] = {
        MTLCompareFunctionNever, MTLCompareFunctionLess, MTLCompareFunctionEqual,
        MTLCompareFunctionLessEqual, MTLCompareFunctionGreater,
        MTLCompareFunctionNotEqual, MTLCompareFunctionGreaterEqual,
        MTLCompareFunctionAlways
    };
    desc.depthCompareFunction = depth_test ? funcs[compare_func] : MTLCompareFunctionAlways;

    id<MTLDepthStencilState> state = [device newDepthStencilStateWithDescriptor:desc];

    id<MTLRenderCommandEncoder> encoder = ctx->currentEncoder;
    if (encoder) {
        [encoder setDepthStencilState:state];
    }

    Py_RETURN_NONE;
}
