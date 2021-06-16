# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from Qt.QtGui import QWindow, QSurface

class GraphicsWindow(QWindow):
    """
    The graphics window that displays the three-dimensional models.
    """

    def __init__(self, parent, ui, stereo = False, opengl_context = None):
        self.session = ui.session
        self.view = ui.session.main_view

        QWindow.__init__(self)
        from Qt.QtWidgets import QWidget
        self.widget = w = QWidget.createWindowContainer(self, parent)
        w.setAcceptDrops(True)
        self.setSurfaceType(QSurface.OpenGLSurface)

        if opengl_context is None:
            from chimerax.graphics import OpenGLContext
            oc = OpenGLContext(self, ui.primaryScreen(), use_stereo = stereo)
        else:
            from chimerax.graphics import OpenGLError
            try:
                opengl_context.enable_stereo(stereo, window = self)
            except OpenGLError as e:
                from chimerax.core.errors import UserError
                raise UserError(str(e))
            oc = opengl_context

        self.opengl_context = oc
        self.view.initialize_rendering(oc)

        self.popup = Popup(self)        # For display of atom spec balloons

    def event(self, event):
        # QWindow does not have drag and drop methods to handle file dropped on app
        # so we detect the drag and drop events here and pass them to the main window.
        if self.handle_drag_and_drop(event):
            return True
        from Qt.QtCore import QEvent
        if event.type() == QEvent.Show:
            self.session.ui.mouse_modes.set_graphics_window(self)
            self._check_opengl()
        return QWindow.event(self, event)

    def _check_opengl(self):
        r = self.view.render
        log = self.session.logger
        from chimerax.graphics import OpenGLVersionError, OpenGLError
        try:
            mc = r.make_current()
        except (OpenGLVersionError, OpenGLError) as e:
            mc = False
            log.error(str(e))
            self.session.update_loop.block_redraw()	# Avoid further opengl errors
        if mc:
            e = r.check_for_opengl_errors()
            if e:
                msg = 'There was an OpenGL graphics error while starting up.  This is usually a problem with the system graphics driver, and the only way to remedy it is to update the graphics driver. ChimeraX will probably not function correctly with the current graphics driver.'
                msg += '\n\n\t"%s"' % e
                log.error(msg)

            self._check_for_bad_intel_driver()

    def _check_for_bad_intel_driver(self):
        import sys
        if sys.platform != 'win32':
            return
        from chimerax.ui.widgets import HtmlView
        if HtmlView.require_native_window:
            return  # Already applied this fix.
        r = self.view.render
        if r.opengl_vendor() == 'Intel':
            ver = r.opengl_version()
            try:
                build = int(ver.split('.')[-1])
            except Exception:
                return
            if 6708 < build < 8280:
                # This is to work around ChimeraX bug #2537 where the entire
                # GUI becomes blank with some 2019 Intel graphics drivers.

                # TODO: This fix may fail if HtmlView widgets are created
                #       before the graphisc is checked.  Worked in tests on one machine.
                HtmlView.require_native_window = True
                msg = ('Your computer has Intel graphics driver %d with a known bug '
                       'that causes all Qt user interface panels to be blank. '
                       'ChimeraX can partially fix this but may make some panel '
                       'titlebars and edges black.  Hopefully newer '
                       'Intel graphics drivers will fix this.' % build)
                self.session.logger.warning(msg)
                                            
    def handle_drag_and_drop(self, event):
        from Qt.QtCore import QEvent
        t = event.type()
        ui = self.session.ui
        if hasattr(ui, 'main_window'):
            mw = ui.main_window
            if t == QEvent.DragEnter:
                mw.dragEnterEvent(event)
                return True
            elif t == QEvent.Drop:
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
            return	# main window not yet initialized.
        if not self.is_drawable:
            return	# Window is not yet exposed so can't use opengl

        # Avoid flickering when resizing by drawing immediately.
        from chimerax.graphics import OpenGLVersionError
        try:
            self.session.update_loop.update_graphics_now()
        except OpenGLVersionError as e:
            # Inadequate OpenGL version
            self.session.logger.error(str(e))

    @property
    def is_drawable(self):
        '''
        Whether graphics window can be drawn to using OpenGL.
        False until window has been created by the native window toolkit.
        '''
        return self.isExposed()
        
    def exposeEvent(self, event):
        self.view.redraw_needed = True

from Qt.QtWidgets import QLabel
class Popup(QLabel):

    def __init__(self, graphics_window):
        from Qt.QtCore import Qt
        QLabel.__init__(self)
        import sys
        if sys.platform == 'darwin':
            # Don't use a Qt.ToolTip which can do undesired auto-hiding on Mac. Bug #2140
            # But on Linux these flags cause huge non-updating balloons, and on Windows
            # the balloon takes the focus (Qt 5.12.4).
            # These flags also cause problems on Mac if ChimeraX is fullscreen, the
            # balloon replaces the entire gui, ChimeraX bug #2210.
#            win_flags = Qt.FramelessWindowHint | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus
            win_flags = Qt.ToolTip
        else:
            win_flags = Qt.ToolTip
        self.setWindowFlags(self.windowFlags() | win_flags)
        self.graphics_window = graphics_window

    def show_text(self, text, position):
        self.setText(text)
        from Qt.QtCore import QPoint
        self.move(self.graphics_window.mapToGlobal(QPoint(*position)))
        self.show()
