#include <Python.h>
#include "vulkan_api.h"

namespace Vulkan_Cpp
{

static struct PyMethodDef vulkan_methods[] =
{
    // Context
    {"create_instance", (PyCFunction)vulkan_create_instance, METH_VARARGS|METH_KEYWORDS, NULL},
    {"create_context", (PyCFunction)vulkan_create_context, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_context", (PyCFunction)vulkan_destroy_context, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_surface_for_window", (PyCFunction)vulkan_set_surface_for_window, METH_VARARGS|METH_KEYWORDS, NULL},
    {"release_window", (PyCFunction)vulkan_release_window, METH_VARARGS|METH_KEYWORDS, NULL},
    {"resize_drawable", (PyCFunction)vulkan_resize_drawable, METH_VARARGS|METH_KEYWORDS, NULL},
    {"device_name", (PyCFunction)vulkan_device_name, METH_VARARGS|METH_KEYWORDS, NULL},
    {"max_3d_texture_size", (PyCFunction)vulkan_max_3d_texture_size, METH_VARARGS|METH_KEYWORDS, NULL},

    // Frame
    {"begin_frame", (PyCFunction)vulkan_begin_frame, METH_VARARGS|METH_KEYWORDS, NULL},
    {"end_frame", (PyCFunction)vulkan_end_frame, METH_VARARGS|METH_KEYWORDS, NULL},

    // Render passes
    {"begin_render_pass", (PyCFunction)vulkan_begin_render_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"begin_depth_pass", (PyCFunction)vulkan_begin_depth_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"end_render_pass", (PyCFunction)vulkan_end_render_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"begin_color_pass", (PyCFunction)vulkan_begin_color_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"begin_mask_pass", (PyCFunction)vulkan_begin_mask_pass, METH_VARARGS|METH_KEYWORDS, NULL},

    // Pipelines
    {"create_pipeline", (PyCFunction)vulkan_create_pipeline, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_pipeline", (PyCFunction)vulkan_set_pipeline, METH_VARARGS|METH_KEYWORDS, NULL},

    // Buffers
    {"create_buffer", (PyCFunction)vulkan_create_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"create_mapped_buffer", (PyCFunction)vulkan_create_mapped_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"update_buffer", (PyCFunction)vulkan_update_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_buffer", (PyCFunction)vulkan_destroy_buffer, METH_VARARGS|METH_KEYWORDS, NULL},

    // Textures
    {"create_texture", (PyCFunction)vulkan_create_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"update_texture", (PyCFunction)vulkan_update_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_texture", (PyCFunction)vulkan_destroy_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"texture_format", (PyCFunction)vulkan_texture_format, METH_VARARGS|METH_KEYWORDS, NULL},

    // Draw commands
    {"set_vertex_buffer", (PyCFunction)vulkan_set_vertex_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_descriptor_buffer", (PyCFunction)vulkan_set_descriptor_buffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_fragment_texture", (PyCFunction)vulkan_set_fragment_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"draw_indexed", (PyCFunction)vulkan_draw_indexed, METH_VARARGS|METH_KEYWORDS, NULL},

    // State
    {"set_viewport", (PyCFunction)vulkan_set_viewport, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_depth_stencil_state", (PyCFunction)vulkan_set_depth_stencil_state, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_cull_mode", (PyCFunction)vulkan_set_cull_mode, METH_VARARGS|METH_KEYWORDS, NULL},

    // Offscreen framebuffer (image capture)
    {"create_offscreen_framebuffer", (PyCFunction)vulkan_create_offscreen_framebuffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_offscreen_framebuffer", (PyCFunction)vulkan_destroy_offscreen_framebuffer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"set_active_offscreen", (PyCFunction)vulkan_set_active_offscreen, METH_VARARGS|METH_KEYWORDS, NULL},
    {"read_framebuffer", (PyCFunction)vulkan_read_framebuffer, METH_VARARGS|METH_KEYWORDS, NULL},

    // Query
    {"get_depth_texture", (PyCFunction)vulkan_get_depth_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"copy_depth_texture", (PyCFunction)vulkan_copy_depth_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"supports_ray_tracing", (PyCFunction)vulkan_supports_ray_tracing, METH_VARARGS|METH_KEYWORDS, NULL},
    {"flush_gpu", (PyCFunction)vulkan_flush_gpu, METH_VARARGS|METH_KEYWORDS, NULL},
    {"supports_metal_objects", (PyCFunction)vulkan_supports_metal_objects, METH_VARARGS|METH_KEYWORDS, NULL},
    {"get_buffer_device_memory", (PyCFunction)vulkan_get_buffer_device_memory, METH_VARARGS|METH_KEYWORDS, NULL},

    // OIT
    {"create_oit_resources", (PyCFunction)vulkan_create_oit_resources, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_oit_resources", (PyCFunction)vulkan_destroy_oit_resources, METH_VARARGS|METH_KEYWORDS, NULL},
    {"begin_oit_pass", (PyCFunction)vulkan_begin_oit_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"end_oit_pass", (PyCFunction)vulkan_end_oit_pass, METH_VARARGS|METH_KEYWORDS, NULL},
    {"get_oit_textures", (PyCFunction)vulkan_get_oit_textures, METH_VARARGS|METH_KEYWORDS, NULL},

    // Ray tracing
    {"build_sphere_accel", (PyCFunction)vulkan_build_sphere_accel, METH_VARARGS|METH_KEYWORDS, NULL},
    {"build_mixed_accel", (PyCFunction)vulkan_build_mixed_accel, METH_VARARGS|METH_KEYWORDS, NULL},
    {"build_scene_accel", (PyCFunction)vulkan_build_scene_accel, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_accel_struct", (PyCFunction)vulkan_destroy_accel_struct, METH_VARARGS|METH_KEYWORDS, NULL},
    {"compile_rtao", (PyCFunction)vulkan_compile_rtao, METH_VARARGS|METH_KEYWORDS, NULL},
    {"dispatch_rtao", (PyCFunction)vulkan_dispatch_rtao, METH_VARARGS|METH_KEYWORDS, NULL},
    {"compile_pathtracer", (PyCFunction)vulkan_compile_pathtracer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"dispatch_pathtracer", (PyCFunction)vulkan_dispatch_pathtracer, METH_VARARGS|METH_KEYWORDS, NULL},

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

extern "C" PyMODINIT_FUNC
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
