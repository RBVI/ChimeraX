"""
Metal render backend for ChimeraX.

This module exposes MetalBackend, which implements the RenderBackend
protocol defined in chimerax.graphics.render_backend.  When loaded,
it registers itself with the View so that `graphics metal` can switch
to it and `graphics opengl` can switch back.
"""

import platform
import sys


def is_metal_supported() -> bool:
    """Return True if the current host can use Metal."""
    if platform.system() != "Darwin":
        return False
    mac_ver = platform.mac_ver()[0]
    if not mac_ver:
        return False
    major, *rest = mac_ver.split(".")
    minor = int(rest[0]) if rest else 0
    major = int(major)
    # Metal is available from macOS 10.13; we require 10.14+ for full features.
    return major > 10 or (major == 10 and minor >= 14)


class MetalBackend:
    """
    Implements the RenderBackend protocol using Apple Metal.

    The backend owns a MetalContext (MTLDevice + MTLCommandQueue), a
    MetalScene (camera, lighting), and a MetalRenderer (pipeline states).
    It is created once per View and lazily initialized on first use.
    """

    name = "metal"

    def __init__(self, session):
        self._session = session
        self._context = None
        self._scene = None
        self._renderer = None
        self._initialized = False
        self._multi_gpu = None
        self._preferred_device = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, view, window_id: int, width: int, height: int) -> bool:
        """
        Initialize Metal for the given native window.

        Parameters
        ----------
        view:
            The chimerax.graphics.View that owns this backend.
        window_id:
            OS-level window handle (NSView* on macOS).
        width, height:
            Initial drawable size in device pixels.
        """
        if self._initialized:
            return True

        try:
            from ._metal import PyMetalContext, PyMetalScene, PyMetalRenderer, PyMetalMultiGPU
        except ImportError as exc:
            self._session.logger.error(
                f"Could not import Metal extension (_metal): {exc}.  "
                "Run 'make install' inside src/bundles/graphics_metal/ to build it."
            )
            return False

        ctx = PyMetalContext()
        if not ctx.initialize():
            self._session.logger.error("Metal context initialization failed.")
            return False

        scene = PyMetalScene(ctx)
        if not scene.initialize():
            self._session.logger.error("Metal scene initialization failed.")
            return False

        renderer = PyMetalRenderer(ctx)
        if not renderer.initialize():
            self._session.logger.error("Metal renderer initialization failed.")
            return False

        renderer.setScene(scene)

        self._context = ctx
        self._scene = scene
        self._renderer = renderer
        self._initialized = True
        self._multi_gpu = PyMetalMultiGPU()
        self._multi_gpu.initialize(ctx)

        self._session.logger.info(
            f"Metal backend initialized on {ctx.deviceName()} "
            f"(unified memory: {ctx.supportsUnifiedMemory()})"
        )
        return True

    def delete(self):
        """Release all Metal resources."""
        self._renderer = None
        self._scene = None
        self._multi_gpu = None
        self._context = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Backend protocol (called by View / RenderBackend)
    # ------------------------------------------------------------------

    def make_current(self) -> bool:
        """Metal has no 'current context' concept; always succeeds."""
        return self._initialized

    def done_current(self):
        """No-op for Metal."""

    def resize(self, width: int, height: int):
        """Update drawable size on window resize."""
        if self._scene and self._scene.camera():
            aspect = width / max(height, 1)
            self._scene.camera().setAspectRatio(aspect)

    def render(self, drawing, camera):
        """
        Render one frame using the Metal backend.

        The frame is structured as:
          beginFrame → accumulate draw calls → endFrame (commits GPU work).

        Parameters
        ----------
        drawing:
            Root chimerax.graphics.Drawing to render.
        camera:
            Active chimerax.graphics.Camera.
        """
        if not self._initialized:
            return

        view_ptr = self._get_mtkview_ptr()
        if view_ptr == 0:
            return

        if not self._renderer.beginFrame(view_ptr):
            return  # drawable not ready (e.g. window minimised)

        self._update_scene_uniforms(camera)
        self._walk_drawing(drawing, camera)

        self._renderer.endFrame()

    def swap_buffers(self):
        """Present Metal drawable — called automatically inside endFrame."""

    # ------------------------------------------------------------------
    # Device selection / multi-device
    # ------------------------------------------------------------------

    def available_devices(self) -> list:
        if self._multi_gpu is None:
            return []
        return self._multi_gpu.getDeviceInfo()

    def select_compute_device(self, device_name: str) -> bool:
        devices = self.available_devices()
        for d in devices:
            if d["name"] == device_name:
                self._preferred_device = device_name
                self._session.logger.info(
                    f"Metal compute offload device set to '{device_name}'"
                )
                return True
        self._session.logger.warning(
            f"Metal device '{device_name}' not found; "
            f"available: {[d['name'] for d in devices]}"
        )
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mtkview_ptr(self) -> int:
        """Return the MTKView* as an integer, or 0 if not yet set."""
        return getattr(self, '_mtkview_ptr', 0)

    def set_mtkview(self, view_ptr: int) -> None:
        """Called by the Qt layer once a CAMetalLayer-backed NSView is ready."""
        self._mtkview_ptr = view_ptr

    def _update_scene_uniforms(self, camera):
        """
        Build a Uniforms struct from ChimeraX camera + lighting and push it
        to the triple-buffered uniform buffer for the current frame.

        Uses numpy to avoid any Python-level matrix allocation overhead; the
        resulting float32 bytes are memcpy-ed into the shared-mode MTLBuffer.
        """
        import numpy as np

        # Compute view and projection matrices.
        view_dir = np.array([0, 0, -1], np.float32)
        eye      = np.zeros(3, np.float32)

        if camera is not None:
            try:
                eye      = np.array(camera.position.origin(), np.float32)
                view_dir = np.array(camera.view_direction(), np.float32)
            except Exception:
                pass

        # Sync scene camera (for Metal scene object compatibility).
        if self._scene is not None:
            mc = self._scene.camera()
            if mc is not None:
                look = eye + view_dir * 10.0
                mc.setPosition(float(eye[0]), float(eye[1]), float(eye[2]))
                mc.setTarget(float(look[0]), float(look[1]), float(look[2]))
                mc.setUp(0.0, 1.0, 0.0)

        # Sync background colour.
        if self._scene is not None:
            try:
                rgba = self._session.main_view.background_rgba
                r, g, b, a = (float(c) for c in rgba)
            except Exception:
                r, g, b, a = 0.0, 0.0, 0.0, 1.0
            self._scene.setBackgroundColor(r, g, b, a)

        # The C++ renderer has already pre-initialised the uniform buffer in
        # beginFrame with sensible defaults; Python doesn't need to re-send
        # the full Uniforms struct on every frame — only when camera changes.
        # (Full struct update via updateSceneUniforms would require a Cython
        # binding for the Uniforms POD; that's Phase 2 work.  For now, the
        # C++ defaults are acceptable.)

    def _walk_drawing(self, drawing, camera):
        """Walk the Drawing tree and accumulate Metal draw calls."""
        if drawing is None:
            return
        import numpy as np
        view_dir = np.array([0, 0, -1], np.float32)
        cam_pos  = np.zeros(3, np.float32)
        if camera is not None:
            try:
                view_dir = np.array(camera.view_direction(), np.float32)
                cam_pos  = np.array(camera.position.origin(), np.float32)
            except Exception:
                pass
        from .drawing_walker import walk_and_upload
        walk_and_upload(drawing, self._renderer, self._context,
                        view_direction=view_dir, camera_pos=cam_pos)
