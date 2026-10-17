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
PyObject* vulkan_create_instance(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_create_context(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_context(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_surface_for_window(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_release_window(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_resize_drawable(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_wait_for_vsync(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_device_name(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_max_3d_texture_size(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_device_info(PyObject* self, PyObject* args, PyObject* kwds);

// Frame lifecycle
PyObject* vulkan_begin_frame(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_end_frame(PyObject* self, PyObject* args, PyObject* kwds);

// Render pass management
PyObject* vulkan_begin_render_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_begin_depth_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_end_render_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_begin_color_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_begin_mask_pass(PyObject* self, PyObject* args, PyObject* kwds);

// Pipeline (shader) management
PyObject* vulkan_create_pipeline(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_pipeline(PyObject* self, PyObject* args, PyObject* kwds);

// Buffer management
PyObject* vulkan_create_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_update_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_buffer(PyObject* self, PyObject* args, PyObject* kwds);

// Texture management
PyObject* vulkan_create_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_update_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_read_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_texture_format(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_texture_identity(PyObject* self, PyObject* args, PyObject* kwds);

// Draw commands
PyObject* vulkan_set_vertex_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_descriptor_buffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_fragment_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_draw_indexed(PyObject* self, PyObject* args, PyObject* kwds);

// Render state
PyObject* vulkan_set_viewport(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_depth_stencil_state(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_cull_mode(PyObject* self, PyObject* args, PyObject* kwds);

// Offscreen framebuffer (image capture)
PyObject* vulkan_create_offscreen_framebuffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_offscreen_framebuffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_set_active_offscreen(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_read_framebuffer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_read_screen(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_get_offscreen_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_copy_framebuffer_to_image(PyObject* self, PyObject* args, PyObject* kwds);

// Query
PyObject* vulkan_get_depth_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_copy_depth_texture(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_depth_texture_size(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_swapchain_state(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_supports_ray_tracing(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_flush_gpu(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_get_buffer_device_memory(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_supports_metal_objects(PyObject* self, PyObject* args, PyObject* kwds);

// Mapped buffers (zero-copy surface upload)
PyObject* vulkan_create_mapped_buffer(PyObject* self, PyObject* args, PyObject* kwds);

// OIT (Order-Independent Transparency; see vulkan_oit.cpp)
PyObject* vulkan_create_oit_resources(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_oit_resources(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_begin_oit_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_end_oit_pass(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_get_oit_textures(PyObject* self, PyObject* args, PyObject* kwds);

// Ray tracing (RTAO + path tracer; see vulkan_raytracing.cpp)
PyObject* vulkan_build_sphere_accel(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_build_mixed_accel(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_build_scene_accel(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_destroy_accel_struct(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_compile_rtao(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_dispatch_rtao(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_compile_pathtracer(PyObject* self, PyObject* args, PyObject* kwds);
PyObject* vulkan_dispatch_pathtracer(PyObject* self, PyObject* args, PyObject* kwds);

#ifdef __cplusplus
}
#endif

#endif // CHIMERAX_VULKAN_API_H
