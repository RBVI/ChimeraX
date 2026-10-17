"""WGPU adapter for the ChimeraX renderer abstraction layer."""

from __future__ import annotations

import numpy as np


class WgpuContext:
    """Own native WGPU presentation canvases for one RAL device."""

    backend_name = "wgpu"

    def __init__(self, graphics_window, screen=None, use_stereo=False, session=None):
        import sys

        if not (sys.platform == "darwin" or sys.platform.startswith("win")):
            raise RuntimeError(
                "Direct WGPU presentation currently supports macOS and Windows"
            )
        if use_stereo:
            raise RuntimeError("WGPU does not support sequential stereo")
        self.window = graphics_window
        self.session = session
        self._canvases = {}
        self.adapter = None
        self.device = None

    def prepare_window(self, window):
        # Surface type and native-window creation are handled by the registry
        # before a canvas context is requested.
        return None

    def canvas_for(self, window):
        key = int(window.winId())
        canvas = self._canvases.get(key)
        if canvas is None:
            canvas = _NativeCanvas(window)
            self._canvases[key] = canvas
        return canvas

    def make_current(self, window=None):
        return True

    def done_current(self):
        return None

    def pixel_scale(self):
        return float(self.window.devicePixelRatio())

    def delete(self):
        for canvas in tuple(self._canvases.values()):
            canvas.close()
        self._canvases.clear()


class _NativeCanvas:
    """Small direct-presentation canvas implementing wgpu-py's contract."""

    manual_present = True

    def __init__(self, window):
        try:
            import wgpu
        except ImportError as error:
            raise RuntimeError(
                "The WGPU backend needs the optional 'wgpu' Python package"
            ) from error
        self.window = window
        present_info = {"window": int(window.winId()), "vsync": True}
        self._context = wgpu.gpu.get_canvas_context(present_info)
        self.update_physical_size()

    def get_wgpu_context(self):
        return self._context

    def get_physical_size(self):
        ratio = float(self.window.devicePixelRatio())
        return (
            max(0, round(self.window.width() * ratio)),
            max(0, round(self.window.height() * ratio)),
        )

    def update_physical_size(self):
        size = self.get_physical_size()
        self._context.set_physical_size(*size)
        return size

    def close(self):
        unconfigure = getattr(self._context, "unconfigure", None)
        if unconfigure is not None:
            unconfigure()


class _UpdateFlag:
    multishadow_update_needed = True

    def max_multishadows(self):
        return 64


class WgpuRender:
    """Scene-data renderer presented through the normal RAL endpoint API."""

    def __init__(self, context, window):
        from .wgpu_prototype import WgpuMeshRenderer

        self.opengl_context = context  # Temporary compatibility name.
        self._context = context
        self._window = window
        self._canvas = context.canvas_for(window)
        self._renderer = WgpuMeshRenderer(
            self._canvas,
            use_render_bundle=True,
            adapter=context.adapter,
            device=context.device,
        )
        if context.adapter is None:
            context.adapter = self._renderer.adapter
            context.device = self._renderer.device
        self._scene_loaded = False
        self._scene_identity = None
        self._view = None
        self._model_handlers = []
        self.multishadow = _UpdateFlag()
        self.front_buffer_valid = False
        self.check_uniforms = False

    @property
    def adapter_summary(self):
        return self._renderer.adapter_summary

    def make_current(self):
        return True

    def done_current(self):
        return None

    def set_target_window(self, window):
        previous = self._window
        if window is not previous:
            self._window = window
            self._canvas = self._context.canvas_for(window)
        return previous

    def release_window(self, window):
        return None

    def delete(self):
        for handler in self._model_handlers:
            handler.remove()
        self._model_handlers.clear()
        self._scene_loaded = False

    def pixel_scale(self):
        return float(self._window.devicePixelRatio())

    def set_default_framebuffer_size(self, width, height):
        self._canvas.update_physical_size()

    def render_size(self):
        return self._canvas.get_physical_size()

    def finish_rendering(self):
        # Deliberately avoid queue.on_submitted_work_done_sync(): older
        # wgpu-native callback ABIs disagree with Python 3.14's cffi wrapper.
        return None

    def frame_buffer_image(self, width, height, front_buffer=False):
        raise RuntimeError("WGPU framebuffer readback is not implemented")

    def opengl_version(self):
        return "WebGPU via wgpu-native"

    def opengl_version_number(self):
        return 0

    def opengl_vendor(self):
        info = getattr(self._renderer.adapter, "info", {})
        return str(info.get("vendor", "WebGPU")) if isinstance(info, dict) else "WebGPU"

    def opengl_renderer(self):
        return self.adapter_summary

    def opengl_info(self):
        return "WGPU adapter: %s" % self.adapter_summary

    def flush_shader_cache(self):
        self._renderer._render_bundle = None
        self._scene_loaded = False

    def draw_view(
        self, view, camera=None, drawings=None,
        check_for_changes=True, swap_buffers=True,
    ):
        """Compile ChimeraX Drawings to WGPU scene records and present."""

        self._attach_view(view)
        drawing_manager = view._drawing_manager
        scene_changed = (
            not self._scene_loaded
            or drawing_manager.redraw_needed
            or bool(getattr(view, "_drawing_changed", False))
        )
        if check_for_changes:
            view.check_for_drawing_change()

        root = view.drawing
        visible = tuple(
            (id(d), bool(getattr(d, "display", True)))
            for d in root.all_drawings(displayed_only=False)
        )
        if visible != self._scene_identity:
            scene_changed = True
            self._scene_identity = visible

        renderer = self._renderer
        self._canvas.update_physical_size()
        if scene_changed:
            from .wgpu_prototype import (
                displayed_highlights,
                displayed_meshes,
                displayed_volumes,
            )

            meshes = displayed_meshes(root)
            if meshes:
                renderer.upload_meshes(meshes)
            else:
                renderer.clear_mesh()
            renderer.upload_highlights(displayed_highlights(root))
            renderer.upload_volumes(displayed_volumes(root))
            self._scene_loaded = True

        from .wgpu_prototype import chimerax_view_projection

        width, height = view.window_size
        if width > 0 and height > 0:
            renderer.set_view_projection(
                chimerax_view_projection(view, (width, height))
            )
        renderer.set_lighting(view)
        renderer.set_silhouette(view.silhouette, self._canvas.get_physical_size())
        renderer.set_highlight(view.highlight_color, view.highlight_thickness)
        renderer.set_background_color(view.background_color)
        self._set_drag_rectangle(view)
        renderer.draw_frame()
        view.update_lighting = False
        view.redraw_needed = False

    def _attach_view(self, view):
        if self._view is view:
            return
        self._view = view
        session = getattr(self._context, "session", None)
        if session is None:
            return
        from chimerax.core.models import (
            ADD_MODELS, MODEL_DISPLAY_CHANGED, REMOVE_MODELS,
        )
        self._model_handlers = [
            session.triggers.add_handler(trigger, self._models_changed)
            for trigger in (ADD_MODELS, REMOVE_MODELS, MODEL_DISPLAY_CHANGED)
        ]

    def _models_changed(self, trigger_name, models):
        self._scene_loaded = False
        if self._view is not None:
            self._view.redraw_needed = True

    def _set_drag_rectangle(self, view):
        rectangle = getattr(view, "_overlay_rectangle", None)
        if rectangle is None:
            self._renderer.set_drag_rectangle(None)
            return
        (x1, y1, x2, y2), color = rectangle
        scale = float(self._window.devicePixelRatio())
        height = self._window.height()
        rect = (
            min(x1, x2) * scale,
            (height - max(y1, y2)) * scale,
            max(x1, x2) * scale,
            (height - min(y1, y2)) * scale,
        )
        self._renderer.set_drag_rectangle(rect, color)


class WgpuTexture:
    """Native-cache record for NumPy textures consumed by WgpuRender."""

    def __init__(self, logical):
        self.data = logical.data
        self.dimension = logical.dimension
        self.size = logical.size
        self.is_cubemap = logical.is_cubemap
        self.linear_interpolation = logical.linear_interpolation
        self.id = None
        self._handle = None

    def initialize_rgba(self, size):
        self.size = tuple(size)

    def initialize_8_bit(self, size):
        self.size = tuple(size)

    def initialize_depth(self, size, depth_compare_mode=True):
        self.size = tuple(size)

    def reload_texture(self, data, now=False):
        self.data = data

    def bind_texture(self, *args, **kwargs):
        return None

    def unbind_texture(self, *args, **kwargs):
        return None

    def read_texture_data(self, numpy_data_type=None):
        data = np.asarray(self.data)
        return data if numpy_data_type is None else data.astype(numpy_data_type)

    def normalization(self):
        dtype = np.asarray(self.data).dtype
        return {
            np.dtype(np.int8): 1 / 127,
            np.dtype(np.uint8): 1 / 255,
            np.dtype(np.int16): 1 / 32767,
            np.dtype(np.uint16): 1 / 65535,
        }.get(dtype, 1.0)

    def delete_texture(self):
        self.data = None
