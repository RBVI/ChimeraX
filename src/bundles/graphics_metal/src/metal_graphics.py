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
        Render one frame.

        Parameters
        ----------
        drawing:
            Root chimerax.graphics.Drawing to render.
        camera:
            Active chimerax.graphics.Camera.
        """
        if not self._initialized:
            return

        self._sync_camera(camera)
        self._sync_background()

        self._renderer.beginFrame()
        self._upload_drawing(drawing)
        self._renderer.endFrame()

    def swap_buffers(self):
        """Present Metal drawable — called automatically by endFrame."""

    # ------------------------------------------------------------------
    # Device selection / multi-device
    # ------------------------------------------------------------------

    def available_devices(self) -> list:
        """Return a list of dicts describing available Metal GPU devices."""
        if self._multi_gpu is None:
            return []
        return self._multi_gpu.getDeviceInfo()

    def select_compute_device(self, device_name: str) -> bool:
        """
        Designate a secondary MTLDevice for async compute offload (e.g.
        volume preprocessing, density analysis).  Returns True on success.

        On Apple Silicon the only Metal device is the integrated GPU; this
        method is meaningful mainly on Intel Macs with discrete + integrated.
        """
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

    def _sync_camera(self, camera):
        """Push ChimeraX camera state into the Metal scene camera."""
        if camera is None or self._scene is None:
            return
        mc = self._scene.camera()
        if mc is None:
            return

        pos = camera.position.origin()
        mc.setPosition(float(pos[0]), float(pos[1]), float(pos[2]))

        fwd = camera.view_direction()
        look = pos + fwd * 10.0
        mc.setTarget(float(look[0]), float(look[1]), float(look[2]))

        up = camera.position.z_axis()
        mc.setUp(float(up[0]), float(up[1]), float(up[2]))

        if hasattr(camera, "field_of_view"):
            mc.setFov(float(camera.field_of_view))
        if hasattr(camera, "near_clip_distance"):
            mc.setNearPlane(float(camera.near_clip_distance))
        if hasattr(camera, "far_clip_distance"):
            mc.setFarPlane(float(camera.far_clip_distance))

    def _sync_background(self):
        """Push the View's background colour into the Metal scene."""
        if self._scene is None:
            return
        try:
            rgba = self._session.main_view.background_rgba
            r, g, b, a = (float(c) for c in rgba)
        except Exception:
            r, g, b, a = 0.0, 0.0, 0.0, 1.0
        self._scene.setBackgroundColor(r, g, b, a)

    def _upload_drawing(self, drawing):
        """
        Walk the Drawing tree and upload fp32 geometry to Metal buffers.

        This is the core integration point.  The DrawingWalker (see
        drawing_walker.py) batches geometry by draw mode and uploads
        MTLBuffer instances that the renderer then encodes.
        """
        if drawing is None:
            return
        from .drawing_walker import walk_and_upload
        walk_and_upload(drawing, self._renderer, self._context)
