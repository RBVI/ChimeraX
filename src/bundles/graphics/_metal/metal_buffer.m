/*
 * Metal buffer management: create, update, destroy GPU buffers.
 */

#include <Python.h>
#include "metal_internal.h"

extern "C"
PyObject* metal_create_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr;
    Py_buffer py_buf;
    static const char* kwlist[] = {"context", "data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ly*"), (char**)kwlist, &ctx_ptr, &py_buf))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;

    id<MTLBuffer> mtl_buf = [ctx->device newBufferWithBytes:py_buf.buf
                                    length:py_buf.len
                                    options:MTLResourceStorageModeShared];
    PyBuffer_Release(&py_buf);

    if (!mtl_buf) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Metal buffer");
        return NULL;
    }

    MetalBuffer* buf = new MetalBuffer();
    buf->buffer = mtl_buf;
    buf->size = py_buf.len;

    return PyLong_FromLong((long)buf);
}

extern "C"
PyObject* metal_update_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long buf_ptr;
    Py_buffer py_buf;
    static const char* kwlist[] = {"buffer", "data", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("ly*"), (char**)kwlist, &buf_ptr, &py_buf))
        return NULL;

    MetalBuffer* buf = (MetalBuffer*)buf_ptr;

    if ((NSUInteger)py_buf.len > buf->size) {
        // Need to reallocate — but we don't have device here
        // For now, error
        PyBuffer_Release(&py_buf);
        PyErr_SetString(PyExc_RuntimeError, "Buffer too small for update; recreate instead");
        return NULL;
    }

    memcpy(buf->buffer.contents, py_buf.buf, py_buf.len);
    PyBuffer_Release(&py_buf);

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_destroy_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long buf_ptr;
    static const char* kwlist[] = {"buffer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("l"), (char**)kwlist, &buf_ptr))
        return NULL;

    MetalBuffer* buf = (MetalBuffer*)buf_ptr;
    buf->buffer = nil;
    delete buf;

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_set_vertex_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, buf_ptr;
    int index, offset;
    static const char* kwlist[] = {"context", "buffer", "index", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llii"), (char**)kwlist,
            &ctx_ptr, &buf_ptr, &index, &offset))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    MetalBuffer* buf = (MetalBuffer*)buf_ptr;

    if (ctx->currentEncoder) {
        [ctx->currentEncoder setVertexBuffer:buf->buffer offset:offset atIndex:index];
    }

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_set_fragment_buffer(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, buf_ptr;
    int index, offset;
    static const char* kwlist[] = {"context", "buffer", "index", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("llii"), (char**)kwlist,
            &ctx_ptr, &buf_ptr, &index, &offset))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    MetalBuffer* buf = (MetalBuffer*)buf_ptr;

    if (ctx->currentEncoder) {
        [ctx->currentEncoder setFragmentBuffer:buf->buffer offset:offset atIndex:index];
    }

    Py_RETURN_NONE;
}

extern "C"
PyObject* metal_draw_indexed(PyObject* self, PyObject* args, PyObject* kwds) {
    long ctx_ptr, index_buf_ptr;
    int primitive_type, index_count, index_offset, instance_count;
    static const char* kwlist[] = {"context", "index_buffer", "primitive_type",
                                   "index_count", "index_offset", "instance_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            const_cast<char*>("lliiil"), (char**)kwlist,
            &ctx_ptr, &index_buf_ptr, &primitive_type,
            &index_count, &index_offset, &instance_count))
        return NULL;

    MetalContext* ctx = (MetalContext*)ctx_ptr;
    MetalBuffer* idx_buf = (MetalBuffer*)index_buf_ptr;

    if (!ctx->currentEncoder)
        Py_RETURN_NONE;

    MTLPrimitiveType prim;
    switch (primitive_type) {
        case 0: prim = MTLPrimitiveTypeTriangle; break;
        case 1: prim = MTLPrimitiveTypeLine; break;
        case 2: prim = MTLPrimitiveTypePoint; break;
        default: prim = MTLPrimitiveTypeTriangle; break;
    }

    [ctx->currentEncoder drawIndexedPrimitives:prim
                         indexCount:index_count
                         indexType:MTLIndexTypeUInt32
                         indexBuffer:idx_buf->buffer
                         indexBufferOffset:index_offset * sizeof(uint32_t)
                         instanceCount:instance_count];

    Py_RETURN_NONE;
}
