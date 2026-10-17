# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from Qt.QtGui import QWindow, QSurface


def _setup_moltenvk_icd():
    """Point the Vulkan loader at the bundled MoltenVK before libvulkan loads.

    Writes a temporary ICD JSON with an absolute path to libMoltenVK.dylib
    so dlopen succeeds without DYLD_LIBRARY_PATH (which SIP strips).
    """
    import os, json, tempfile
    lib_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "graphics", "src", "lib"
    )
    lib_dir = os.path.normpath(lib_dir)
    mvk_path = os.path.join(lib_dir, "libMoltenVK.dylib")
    if not os.path.exists(mvk_path):
        return  # Not bundled; rely on system Vulkan SDK
    icd = {
        "file_format_version": "1.0.0",
        "ICD": {
            "library_path": mvk_path,
            "api_version": "1.4.0",
            "is_portability_driver": True,
        },
    }
    fd, icd_path = tempfile.mkstemp(suffix=".json", prefix="chimerax_mvk_")
    with os.fdopen(fd, "w") as f:
        json.dump(icd, f)
    os.environ["VK_DRIVER_FILES"] = icd_path
    os.environ["VK_ICD_FILENAMES"] = icd_path
    import sys
    print(f"[MoltenVK] ICD={icd_path} MVK={mvk_path} exists={os.path.exists(mvk_path)}", file=sys.stderr, flush=True)
    print(f"[MoltenVK] VK_DRIVER_FILES={os.environ.get('VK_DRIVER_FILES')}", file=sys.stderr, flush=True)
    with open(icd_path) as _f:
        print(f"[MoltenVK] JSON contents: {_f.read()}", file=sys.stderr, flush=True)


# Must run at module import time, before any Vulkan import loads libvulkan.
import os as _os, sys as _sys
_moltenvk_icd_initialized = False
if _sys.platform == "darwin" and _os.environ.get("CHIMERAX_GRAPHICS_BACKEND", "").lower() == "vulkan":
    _setup_moltenvk_icd()
    _moltenvk_icd_initialized = True
del _os, _sys


def _ensure_moltenvk_icd():
    '''Idempotent guard around _setup_moltenvk_icd(). Used when switching to
    Vulkan at runtime on macOS — the ICD must be configured before libvulkan
    is loaded, which has not yet happened if startup backend was OpenGL.'''
    global _moltenvk_icd_initialized
    import sys
    if _moltenvk_icd_initialized or sys.platform != "darwin":
        return
    _setup_moltenvk_icd()
    _moltenvk_icd_initialized = True


class GraphicsWindow(QWindow):
    """
    The graphics window that displays the three-dimensional models.
    """

    def __init__(self, parent, ui, stereo=False, opengl_context=None):
        self.session = ui.session
        self.view = ui.session.main_view

        QWindow.__init__(self)
        from Qt.QtWidgets import QWidget

        self.widget = w = QWidget.createWindowContainer(self, parent)
        w.setAcceptDrops(True)
        import os, sys

        self._ui = ui
        self._stereo = stereo
        backend = os.environ.get("CHIMERAX_GRAPHICS_BACKEND", "").lower()
        if backend == "vulkan":
            oc = self._create_context("vulkan")
        elif backend == "wgpu":
            oc = self._create_context("wgpu")
        elif sys.platform == "darwin" and backend == "metal":
            oc = self._create_context("metal")
        elif opengl_context is None:
            oc = self._create_context("opengl")
        else:
            self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
            from chimerax.graphics import OpenGLError

            try:
                opengl_context.enable_stereo(stereo, window=self)
            except OpenGLError as e:
                from chimerax.core.errors import UserError

                raise UserError(str(e))
            oc = opengl_context
            from chimerax.graphics.ral import (
                GraphicsDevice,
                backend_registry,
            )

            device = GraphicsDevice(
                backend_registry.get("opengl"), opengl_context
            )
            opengl_context.graphics_device = device
            self.graphics_device = device

        self.opengl_context = oc
        self.view.initialize_rendering(oc)
        self.render_endpoint = self.view.render_endpoint

        self.popup = Popup(self)  # For display of atom spec balloons

    def _create_context(self, backend):
        '''Construct a graphics context for the requested backend, setting the
        appropriate QSurface.SurfaceType on this QWindow first. Returns the
        new context object; does not assign it.'''
        import sys

        if backend == "vulkan" and sys.platform == "darwin":
            _ensure_moltenvk_icd()
        from chimerax.graphics.ral import GraphicsDevice

        try:
            device = GraphicsDevice.create(
                backend, self, self._ui, stereo=self._stereo
            )
        except ValueError as error:
            from chimerax.core.errors import UserError

            raise UserError(str(error)) from error
        self.graphics_device = device
        return device.context

    def _release_graphics_backend(self):
        '''Release the current endpoint and device, preserving scene data.'''
        self.view.release_graphics_context()
        device = getattr(self, "graphics_device", None)
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        else:
            context = getattr(self, "opengl_context", None)
            if context is not None and hasattr(context, "delete"):
                try:
                    context.delete()
                except Exception:
                    pass
        self.graphics_device = None
        self.opengl_context = None
        self.render_endpoint = None

    def _recreate_native_window(self, backend, make_visible=False):
        '''Recreate the platform window before selecting a new surface type.

        In particular, Qt must create the macOS NSView as a Metal surface;
        changing an already-created OpenGL window to Metal leaves Vulkan with
        a surface for which MoltenVK cannot create a swapchain.
        '''
        self.destroy()
        self._exposed = False
        if hasattr(self, "_chimerax_vulkan_instance"):
            self._chimerax_vulkan_instance = None
        # Configure the QWindow before anything (including show() or winId())
        # can recreate its platform window.  On macOS Qt does not reliably
        # install the CAMetalLayer for an unexposed embedded QWindow, so expose
        # it before MoltenVK asks for the initial swapchain.
        from chimerax.graphics.ral import backend_registry
        backend_registry.get(backend).configure_surface(self, None)
        import sys
        if sys.platform == "darwin" and make_visible:
            self.show()
        # Touch delivery is enabled on the native view (see
        # MultitouchTrackpad.set_graphics_window), and destroy() replaced
        # that view, so trackpad gestures stop working unless re-enabled.
        # Before the first show, the Show event handler does this instead.
        if getattr(self, "_first_show", False):
            self.session.ui.mouse_modes.trackpad.set_graphics_window(self)

    def _install_graphics_backend(self, backend):
        '''Create and attach one backend to this window.'''
        context = self._create_context(backend)
        self.opengl_context = context
        self.view.initialize_rendering(context)
        self.render_endpoint = self.view.render_endpoint

    def switch_backend(self, backend, *, force=False, vulkan_runtime=None):
        '''Swap the active graphics backend in place. The scene is preserved;
        per-context GPU resources are released and re-uploaded on next draw.
        Allowed values: "opengl", "vulkan", "wgpu".'''
        backend = backend.lower()
        if backend not in ("opengl", "vulkan", "wgpu"):
            from chimerax.core.errors import UserError
            raise UserError(
                f"graphics backend must be opengl, vulkan, or wgpu, got {backend!r}"
            )
        endpoint = getattr(self, "render_endpoint", None)
        if endpoint is not None:
            old_backend = endpoint.backend_name
        else:
            context = getattr(self, "opengl_context", None)
            old_backend = getattr(context, "backend_name", "opengl")
        if backend == old_backend and not force:
            return
        old_runtime = getattr(self, '_vulkan_runtime', None)

        ul = self.session.update_loop
        ul.block_redraw()
        try:
            from chimerax.core.core_triggers import (
                GRAPHICS_BACKEND_CHANGED, GRAPHICS_BACKEND_CHANGING,
            )
            # Auxiliary native windows must release resources while the old
            # context is still alive, then rebuild against the new context.
            self.session.triggers.activate_trigger(
                GRAPHICS_BACKEND_CHANGING, {"backend": backend})
            was_visible = self.isVisible()
            self._release_graphics_backend()
            # Releasing an active headset also destroys its runtime session.
            old_runtime = getattr(self, '_vulkan_runtime', None)
            self._vulkan_runtime = vulkan_runtime if force else None

            # Surface type is a platform-window creation property.  Reusing
            # the QWindow object keeps createWindowContainer and input wiring
            # intact, while destroy() makes Qt build a fresh native child with
            # the target backend's OpenGL/Vulkan/Metal surface type.
            self._recreate_native_window(backend, make_visible=was_visible)
            try:
                self._install_graphics_backend(backend)
            except Exception as switch_error:
                # A backend probe must not strand the application without a
                # renderer.  Recreate the previous backend and notify clients
                # that released auxiliary endpoints during CHANGING.
                self._release_graphics_backend()
                self._vulkan_runtime = old_runtime
                self._recreate_native_window(
                    old_backend, make_visible=was_visible)
                try:
                    self._install_graphics_backend(old_backend)
                except Exception as rollback_error:
                    self._release_graphics_backend()
                    raise RuntimeError(
                        f"Could not switch graphics backend to {backend!r}; "
                        f"restoring {old_backend!r} also failed: "
                        f"{rollback_error}"
                    ) from switch_error
                self.session.triggers.activate_trigger(
                    GRAPHICS_BACKEND_CHANGED, {"backend": old_backend})
                if was_visible:
                    self.show()
                raise

            self.session.triggers.activate_trigger(
                GRAPHICS_BACKEND_CHANGED, {"backend": backend})
            if was_visible:
                self.show()
        finally:
            ul.unblock_redraw()
        # initialize_rendering() marks the view dirty.  Let the normal update
        # timer draw the first backend frame so this command does not block on
        # full scene upload and cold pipeline creation.
        self.view.redraw_needed = True

    def event(self, event):
        # QWindow does not have drag and drop methods to handle file dropped on app
        # so we detect the drag and drop events here and pass them to the main window.
        if self.handle_drag_and_drop(event):
            return True
        from Qt.QtCore import QEvent

        if event.type() == QEvent.Type.Show and not getattr(self, "_first_show", False):
            self._first_show = True
            self.session.ui.mouse_modes.set_graphics_window(self)
            self._check_opengl()
        return QWindow.event(self, event)

    def _check_opengl(self):
        r = self.view.render
        log = self.session.logger
        if getattr(r, "backend_name", "opengl") == "wgpu":
            log.info("WGPU graphics area: %s" % r.adapter_summary)
            return
        from chimerax.graphics import OpenGLVersionError, OpenGLError
        from chimerax.graphics import (
            remember_current_opengl_context,
            restore_current_opengl_context,
        )

        cc = remember_current_opengl_context()
        try:
            mc = r.make_current()
        except (OpenGLVersionError, OpenGLError) as e:
            mc = False
            log.error(str(e))
            self.session.update_loop.block_redraw()  # Avoid further opengl errors
        if mc:
            e = r.check_for_opengl_errors()
            if e:
                msg = "There was an OpenGL graphics error while starting up.  This is usually a problem with the system graphics driver, and the only way to remedy it is to update the graphics driver. ChimeraX will probably not function correctly with the current graphics driver."
                msg += '\n\n\t"%s"' % e
                log.error(msg)

            self._check_for_bad_intel_driver()

        restore_current_opengl_context(cc)

    def _check_for_bad_intel_driver(self):
        import sys

        if sys.platform != "win32":
            return
        from chimerax.ui.widgets import HtmlView

        if HtmlView.require_native_window:
            return  # Already applied this fix.
        r = self.view.render
        if r.opengl_vendor() == "Intel":
            ver = r.opengl_version()
            try:
                build = int(ver.split(".")[-1])
            except Exception:
                return
            if "Build 26.20.100." in ver and 6708 < build < 8280:
                # This is to work around ChimeraX bug #2537 where the entire
                # GUI becomes blank with some 2019 Intel graphics drivers.

                # TODO: This fix may fail if HtmlView widgets are created
                #       before the graphics is checked.  Worked in tests on one machine.
                HtmlView.require_native_window = True
                msg = (
                    "Your computer has Intel graphics driver %d with a known bug "
                    "that causes all Qt user interface panels to be blank. "
                    "ChimeraX can partially fix this but may make some panel "
                    "titlebars and edges black.  Hopefully newer "
                    "Intel graphics drivers will fix this." % build
                )
                self.session.logger.warning(msg)

    def handle_drag_and_drop(self, event):
        from Qt.QtCore import QEvent

        t = event.type()
        ui = self.session.ui
        if hasattr(ui, "main_window"):
            mw = ui.main_window
            if t == QEvent.Type.DragEnter:
                mw.dragEnterEvent(event)
                return True
            elif t == QEvent.Type.Drop:
                mw.dropEvent(event)
                return True

    # Override QWindow size(), width() and height() to use widget values.
    # In Qt 5.12.9 QWindow reports values that are half the correct size
    # after main window is dragged from devicePixelRatio = 2 screen
    # to a devicePixelRatio = 1 screen on Windows 10.
    def size(self):
        return self.widget.size()

    def width(self):
        return self.widget.width()

    def height(self):
        return self.widget.height()

    def resizeEvent(self, event):
        s = self.size()
        w, h = s.width(), s.height()
        v = self.view
        v.resize(w, h)
        v.redraw_needed = True

        if self.session.ui.main_window is None:
            return  # main window not yet initialized.
        if not self.is_drawable:
            return  # Window is not yet exposed so can't use opengl

        # Avoid flickering when resizing by drawing immediately.
        from chimerax.graphics import OpenGLVersionError

        try:
            self.session.update_loop.update_graphics_now()
        except OpenGLVersionError as e:
            # Inadequate OpenGL version
            self.session.logger.error(str(e))

    @property
    def is_drawable(self):
        """
        Whether graphics window can be drawn to using OpenGL.
        False until window has been created by the native window toolkit.
        """
        return self.isExposed()

    def exposeEvent(self, event):
        exposed = self.isExposed()
        newly_exposed = exposed and not getattr(self, "_exposed", False)
        self._exposed = exposed
        if newly_exposed or not self._layer_keeps_contents():
            self.view.redraw_needed = True

    def _layer_keeps_contents(self):
        # On macOS, MoltenVK marks the CAMetalLayer as needing display on
        # every present, so Qt sends an Expose after every Vulkan frame.
        # Redrawing for each one rendered continuously even when nothing
        # changed.  A Metal layer keeps its last frame, so only a newly
        # exposed window needs a redraw; resizes request their own.
        import sys
        if sys.platform != "darwin":
            return False
        endpoint = getattr(self, "render_endpoint", None)
        return endpoint is not None and endpoint.backend_name != "opengl"


from Qt.QtWidgets import QLabel


class Popup(QLabel):
    def __init__(self, graphics_window):
        from Qt.QtCore import Qt

        QLabel.__init__(self)
        import sys

        if sys.platform == "darwin":
            # Don't use a Qt.ToolTip which can do undesired auto-hiding on Mac. Bug #2140
            # But on Linux these flags cause huge non-updating balloons, and on Windows
            # the balloon takes the focus (Qt 5.12.4).
            # These flags also cause problems on Mac if ChimeraX is fullscreen, the
            # balloon replaces the entire gui, ChimeraX bug #2210.
            #            win_flags = Qt.FramelessWindowHint | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus
            win_flags = Qt.WindowType.ToolTip
        else:
            win_flags = Qt.WindowType.ToolTip
        self.setWindowFlags(self.windowFlags() | win_flags)
        self.graphics_window = graphics_window

    def show_text(self, text, position):
        self.setText(text)
        from Qt.QtCore import QPoint

        self.move(self.graphics_window.mapToGlobal(QPoint(*position)))
        self.show()
