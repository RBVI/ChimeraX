/*
 * Vulkan backend C API for ChimeraX graphics.
 *
 * These functions are called from the Python _vulkan extension module.
 * All Vulkan API code is isolated in .cpp files.
 */

#ifndef CHIMERAX_VULKAN_API_H
#define CHIMERAX_VULKAN_API_H

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

// Context management
PyObject* vulkan_create_context(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_context(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_resize_drawable(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_device_name(PyObject* self, PyObject* args, PyObject* kwds);

// Frame lifecycle
PyObject* vulkan_begin_frame(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_end_frame(PyObject* self, PyObject* args, PyObject* kwds);

// Render pass management
PyObject* vulkan_begin_render_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_begin_depth_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_end_render_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_begin_color_pass(PyObject* self, PyObject* args, PyObject* kwds);

// Pipeline (shader) management
PyObject* vulkan_create_pipeline(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_pipeline(PyObject* self, PyObject* args, PyObject* kwds);

// Buffer management
PyObject* vulkan_create_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_update_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_buffer(PyObject* self, PyObject* args, PyObject* kwds);

// Texture management
PyObject* vulkan_create_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_texture(PyObject* self, PyObject* args, PyObject* kwds);

// Draw commands
PyObject* vulkan_set_vertex_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_descriptor_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_fragment_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_draw_indexed(PyObject* self, PyObject* args, PyObject* kwds);

// Render state
PyObject* vulkan_set_viewport(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_depth_stencil_state(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_cull_mode(PyObject* self, PyObject* args, PyObject* kwds);

// Query
PyObject* vulkan_get_depth_texture(PyObject* self, PyObject* args, PyObject* kwds);

#ifdef __cplusplus
}
#endif

#endif // CHIMERAX_VULKAN_API_H
