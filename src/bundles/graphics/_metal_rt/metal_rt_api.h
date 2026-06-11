/*
 * Public C API for the _metal_rt Python extension module.
 * Each function is exposed as a Python method via the module's method table.
 */

#ifndef CHIMERAX_METAL_RT_API_H
#define CHIMERAX_METAL_RT_API_H

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

// macOS surface creation and resize (ObjC — must live here, not in _vulkan)
PyObject* metal_rt_create_surface(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_rt_resize_surface(PyObject* self, PyObject* args, PyObject* kwds);

// Texture interop (VulkanTexture* -> MetalRTTexture*)
PyObject* metal_rt_export_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_rt_release_texture(PyObject* self, PyObject* args, PyObject* kwds);

// Buffer interop (VulkanBuffer* -> MTLBuffer*)
PyObject* metal_rt_export_buffer(PyObject* self, PyObject* args, PyObject* kwds);

// Initialization — extracts MTLDevice/MTLCommandQueue from a VulkanContext
// via VK_EXT_metal_objects.
PyObject* metal_rt_init(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_rt_destroy(PyObject* self, PyObject* args, PyObject* kwds);

// Acceleration structures
PyObject* metal_rt_build_sphere_accel(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_rt_build_mixed_accel(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_rt_build_scene_accel(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_rt_destroy_accel_struct(PyObject* self, PyObject* args, PyObject* kwds);

// RTAO
PyObject* metal_rt_compile_rtao(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_rt_dispatch_rtao(PyObject* self, PyObject* args, PyObject* kwds);

// Path tracer
PyObject* metal_rt_compile_pathtracer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_rt_dispatch_pathtracer(PyObject* self, PyObject* args, PyObject* kwds);

#ifdef __cplusplus
}
#endif

#endif // CHIMERAX_METAL_RT_API_H
