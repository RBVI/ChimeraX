#include <Python.h>
#include "vulkan_api.h"

namespace Vulkan_Cpp
{

static struct PyMethodDef vulkan_methods[] =
{
    // Context
    {"create_context", (PyCFunction)vulkan_create_context, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_context", (PyCFunction)vulkan_destroy_context, METH_VARARGS|METH_KEYWORDS, NULL},
    {"resize_drawable", (PyCFunction)vulkan_resize_drawable, METH_VARARGS|METH_KEYWORDS, NULL},
    {"device_name", (PyCFunction)vulkan_device_name, METH_VARARGS|METH_KEYWORDS, NULL},

    // Frame
    {"begin_frame", (PyCFunction)vulkan_begin_frame, METH_VARARGS|METH_KEYWORDS, NULL},
    {"end_frame", (PyCFunction)vulkan_end_frame, METH_VARARGS|METH_KEYWORDS, NULL},

    // Render passes
    {"begin_render_pass", (PyCFunction)vulkan_begin_render_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"begin_depth_pass", (PyCFunction)vulkan_begin_depth_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"end_render_pass", (PyCFunction)vulkan_end_render_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"begin_color_pass", (PyCFunction)vulkan_begin_color_pass, METH_VARARGS|METH_KEYWORDS, NULL},

    // Pipelines
    {"create_pipeline", (PyCFunction)vulkan_create_pipeline, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_pipeline", (PyCFunction)vulkan_set_pipeline, METH_VARARGS|METH_KEYWORDS, NULL},

    // Buffers
    {"create_buffer", (PyCFunction)vulkan_create_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"update_buffer", (PyCFunction)vulkan_update_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_buffer", (PyCFunction)vulkan_destroy_buffer, METH_VARARGS|METH_KEYWORDS, NULL},

    // Draw commands
    {"set_vertex_buffer", (PyCFunction)vulkan_set_vertex_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_descriptor_buffer", (PyCFunction)vulkan_set_descriptor_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_fragment_texture", (PyCFunction)vulkan_set_fragment_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"draw_indexed", (PyCFunction)vulkan_draw_indexed, METH_VARARGS|METH_KEYWORDS, NULL},

    // State
    {"set_viewport", (PyCFunction)vulkan_set_viewport, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_depth_stencil_state", (PyCFunction)vulkan_set_depth_stencil_state, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_cull_mode", (PyCFunction)vulkan_set_cull_mode, METH_VARARGS|METH_KEYWORDS, NULL},

    // Query
    {"get_depth_texture", (PyCFunction)vulkan_get_depth_texture, METH_VARARGS|METH_KEYWORDS, NULL},

    {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int vulkan_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int vulkan_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_vulkan",
    "ChimeraX Vulkan rendering backend",
    sizeof(struct module_state),
    vulkan_methods,
    NULL,
    vulkan_traverse,
    vulkan_clear,
    NULL
};

extern "C" __attribute__((__visibility__("default"))) PyMODINIT_FUNC
PyInit__vulkan(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("_vulkan.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}

}  // Vulkan_Cpp namespace
