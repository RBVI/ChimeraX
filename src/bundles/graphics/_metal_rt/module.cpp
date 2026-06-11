/*
 * Python extension module for Metal ray tracing via MoltenVK interop.
 */

#include <Python.h>
#include "metal_rt_api.h"

static PyMethodDef metal_rt_methods[] = {
    // Surface creation (macOS / MoltenVK)
    {"create_surface", (PyCFunction)metal_rt_create_surface, METH_VARARGS|METH_KEYWORDS, NULL},
    {"resize_surface", (PyCFunction)metal_rt_resize_surface, METH_VARARGS|METH_KEYWORDS, NULL},

    // Texture interop
    {"export_texture", (PyCFunction)metal_rt_export_texture, METH_VARARGS|METH_KEYWORDS, NULL},
    {"release_texture", (PyCFunction)metal_rt_release_texture, METH_VARARGS|METH_KEYWORDS, NULL},

    // Buffer interop
    {"export_buffer", (PyCFunction)metal_rt_export_buffer, METH_VARARGS|METH_KEYWORDS, NULL},

    // Initialization
    {"init_from_vulkan_context", (PyCFunction)metal_rt_init, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy", (PyCFunction)metal_rt_destroy, METH_VARARGS|METH_KEYWORDS, NULL},

    // Acceleration structures
    {"build_sphere_accel", (PyCFunction)metal_rt_build_sphere_accel, METH_VARARGS|METH_KEYWORDS, NULL},
    {"build_mixed_accel", (PyCFunction)metal_rt_build_mixed_accel, METH_VARARGS|METH_KEYWORDS, NULL},
    {"build_scene_accel", (PyCFunction)metal_rt_build_scene_accel, METH_VARARGS|METH_KEYWORDS, NULL},
    {"destroy_accel_struct", (PyCFunction)metal_rt_destroy_accel_struct, METH_VARARGS|METH_KEYWORDS, NULL},

    // RTAO
    {"compile_rtao", (PyCFunction)metal_rt_compile_rtao, METH_VARARGS|METH_KEYWORDS, NULL},
    {"dispatch_rtao", (PyCFunction)metal_rt_dispatch_rtao, METH_VARARGS|METH_KEYWORDS, NULL},

    // Path tracer
    {"compile_pathtracer", (PyCFunction)metal_rt_compile_pathtracer, METH_VARARGS|METH_KEYWORDS, NULL},
    {"dispatch_pathtracer", (PyCFunction)metal_rt_dispatch_pathtracer, METH_VARARGS|METH_KEYWORDS, NULL},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef metal_rt_module = {
    PyModuleDef_HEAD_INIT,
    "_metal_rt",
    "Metal ray tracing via MoltenVK interop (VK_EXT_metal_objects)",
    -1,
    metal_rt_methods
};

PyMODINIT_FUNC PyInit__metal_rt(void) {
    return PyModule_Create(&metal_rt_module);
}
