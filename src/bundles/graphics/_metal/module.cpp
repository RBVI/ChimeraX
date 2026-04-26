/*
 * Python extension module for Metal rendering backend.
 *
 * Provides low-level Metal API access from Python. The high-level
 * Renderer protocol implementation lives in metal.py.
 */

#include <Python.h>
#include "metal_api.h"

namespace Metal_Cpp
{

static struct PyMethodDef metal_methods[] =
{
    // Context
    {"create_context", (PyCFunction)metal_create_context, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_context", (PyCFunction)metal_destroy_context, METH_VARARGS|METH_KEYWORDS, NULL},
    {"resize_drawable", (PyCFunction)metal_resize_drawable, METH_VARARGS|METH_KEYWORDS, NULL},

    // Frame lifecycle
    {"begin_frame", (PyCFunction)metal_begin_frame, METH_VARARGS|METH_KEYWORDS, NULL},
    {"end_frame", (PyCFunction)metal_end_frame, METH_VARARGS|METH_KEYWORDS, NULL},

    // Render passes
    {"begin_render_pass", (PyCFunction)metal_begin_render_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"begin_depth_pass", (PyCFunction)metal_begin_depth_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"end_render_pass", (PyCFunction)metal_end_render_pass, METH_VARARGS|METH_KEYWORDS, NULL},

    // Pipelines
    {"create_pipeline", (PyCFunction)metal_create_pipeline, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_pipeline", (PyCFunction)metal_set_pipeline, METH_VARARGS|METH_KEYWORDS, NULL},

    // Buffers
    {"create_buffer", (PyCFunction)metal_create_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"update_buffer", (PyCFunction)metal_update_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_buffer", (PyCFunction)metal_destroy_buffer, METH_VARARGS|METH_KEYWORDS, NULL},

    // Textures
    {"create_texture", (PyCFunction)metal_create_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_texture", (PyCFunction)metal_destroy_texture, METH_VARARGS|METH_KEYWORDS, NULL},

    // Draw commands
    {"set_vertex_buffer", (PyCFunction)metal_set_vertex_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_fragment_buffer", (PyCFunction)metal_set_fragment_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_fragment_texture", (PyCFunction)metal_set_fragment_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"draw_indexed", (PyCFunction)metal_draw_indexed, METH_VARARGS|METH_KEYWORDS, NULL},

    // State
    {"set_viewport", (PyCFunction)metal_set_viewport, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_depth_stencil_state", (PyCFunction)metal_set_depth_stencil_state, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_cull_mode", (PyCFunction)metal_set_cull_mode, METH_VARARGS|METH_KEYWORDS, NULL},

    // Query
    {"device_name", (PyCFunction)metal_device_name, METH_VARARGS|METH_KEYWORDS, NULL},
    {"get_depth_texture", (PyCFunction)metal_get_depth_texture, METH_VARARGS|METH_KEYWORDS, NULL},

    // Color render pass (post-processing)
    {"begin_color_pass", (PyCFunction)metal_begin_color_pass, METH_VARARGS|METH_KEYWORDS, NULL},

    // Ray tracing
    {"build_sphere_accel", (PyCFunction)metal_build_sphere_accel, METH_VARARGS|METH_KEYWORDS, NULL},
    {"build_mixed_accel", (PyCFunction)metal_build_mixed_accel, METH_VARARGS|METH_KEYWORDS, NULL},
    {"compile_rtao", (PyCFunction)metal_compile_rtao, METH_VARARGS|METH_KEYWORDS, NULL},
    {"dispatch_rtao", (PyCFunction)metal_dispatch_rtao, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_accel_struct", (PyCFunction)metal_destroy_accel_struct, METH_VARARGS|METH_KEYWORDS, NULL},

    // Path tracer
    {"compile_pathtracer", (PyCFunction)metal_compile_pathtracer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"dispatch_pathtracer", (PyCFunction)metal_dispatch_pathtracer, METH_VARARGS|METH_KEYWORDS, NULL},

    // Multi-window
    {"set_layer_for_view", (PyCFunction)metal_set_layer_for_view, METH_VARARGS|METH_KEYWORDS, NULL},

    {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int metal_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int metal_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_metal",
    "ChimeraX Metal rendering backend",
    sizeof(struct module_state),
    metal_methods,
    NULL,
    metal_traverse,
    metal_clear,
    NULL
};

extern "C" __attribute__((__visibility__("default"))) PyMODINIT_FUNC
PyInit__metal(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("_metal.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}

}  // Metal_Cpp namespace
