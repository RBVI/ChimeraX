/*
 * Metal backend C API for ChimeraX graphics.
 *
 * These functions are called from the Python _metal extension module.
 * All Metal/Objective-C code is isolated in .mm files.
 */

#ifndef CHIMERAX_METAL_API_H
#define CHIMERAX_METAL_API_H

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

// Context management
PyObject* metal_create_context(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_destroy_context(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_resize_drawable(PyObject* self, PyObject* args, PyObject* kwds);

// Frame lifecycle
PyObject* metal_begin_frame(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_end_frame(PyObject* self, PyObject* args, PyObject* kwds);

// Render pass management
PyObject* metal_begin_render_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_begin_depth_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_end_render_pass(PyObject* self, PyObject* args, PyObject* kwds);

// Pipeline (shader) management
PyObject* metal_create_pipeline(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_set_pipeline(PyObject* self, PyObject* args, PyObject* kwds);

// Buffer management
PyObject* metal_create_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_update_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_destroy_buffer(PyObject* self, PyObject* args, PyObject* kwds);

// Texture management
PyObject* metal_create_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_destroy_texture(PyObject* self, PyObject* args, PyObject* kwds);

// Draw commands
PyObject* metal_set_vertex_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_set_fragment_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_set_fragment_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_draw_indexed(PyObject* self, PyObject* args, PyObject* kwds);

// Render state
PyObject* metal_set_viewport(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_set_depth_stencil_state(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_set_cull_mode(PyObject* self, PyObject* args, PyObject* kwds);

// Query
PyObject* metal_device_name(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_get_depth_texture(PyObject* self, PyObject* args, PyObject* kwds);

// Color render pass targeting a specific texture
PyObject* metal_begin_color_pass(PyObject* self, PyObject* args, PyObject* kwds);

// Ray tracing
PyObject* metal_build_sphere_accel(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_build_mixed_accel(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_compile_rtao(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_dispatch_rtao(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_destroy_accel_struct(PyObject* self, PyObject* args, PyObject* kwds);

// Path tracer
PyObject* metal_compile_pathtracer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* metal_dispatch_pathtracer(PyObject* self, PyObject* args, PyObject* kwds);

// Multi-window support
PyObject* metal_set_layer_for_view(PyObject* self, PyObject* args, PyObject* kwds);

#ifdef __cplusplus
}
#endif

#endif // CHIMERAX_METAL_API_H
