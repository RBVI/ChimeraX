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

from PyQt5.QtGui import QWindow, QSurface

class GraphicsWindow(QWindow):
    """
    The graphics window that displays the three-dimensional models.
    """

    def __init__(self, parent, ui, stereo = False, opengl_context = None):
        QWindow.__init__(self)
        from PyQt5.QtWidgets import QWidget
        self.widget = w = QWidget.createWindowContainer(self, parent)
        w.setAcceptDrops(True)
        self.setSurfaceType(QSurface.OpenGLSurface)
        self.timer = None
        self.session = ui.session
        self.view = ui.session.main_view

        self.redraw_interval = 16.667  # milliseconds
        #   perhaps redraw interval should be 10 to reduce
        #   frame drops at 60 frames/sec
        self.minimum_event_processing_ratio = 0.1 # Event processing time as a fraction
        # of time since start of last drawing
        self.last_redraw_start_time = self.last_redraw_finish_time = 0

        if opengl_context is None:
            from chimerax.core.graphics import OpenGLContext
            oc = OpenGLContext(self, ui.primaryScreen(), use_stereo = stereo)
        else:
            from chimerax.core.graphics import OpenGLError
            try:
                opengl_context.enable_stereo(stereo, window = self)
            except OpenGLError as e:
                from chimerax.core.errors import UserError
                raise UserError(str(e))
            oc = opengl_context

        self.opengl_context = oc
        self.view.initialize_rendering(oc)

        self.popup = Popup(self)        # For display of atom spec balloons

        ui.mouse_modes.set_graphics_window(self)

    def event(self, event):
        # QWindow does not have drag and drop methods to handle file dropped on app
        # so we detect the drag and drop events here and pass them to the main window.
        if self.handle_drag_and_drop(event):
            return True
        return QWindow.event(self, event)

    def handle_drag_and_drop(self, event):
        from PyQt5.QtCore import QEvent
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
    
    def resizeEvent(self, event):
        s = event.size()
        w, h = s.width(), s.height()
        v = self.view
        v.resize(w, h)
        v.redraw_needed = True
        if self.is_drawable:
            # Avoid flickering when resizing by drawing immediately.
            from chimerax.core.graphics import OpenGLVersionError
            try:
                v.draw(check_for_changes = False)
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
        
    def set_redraw_interval(self, msec):
        self.redraw_interval = msec  # milliseconds
        t = self.timer
        if t is not None:
            t.start(self.redraw_interval)

    def start_redraw_timer(self):
        if self.timer is not None:
            return
        from PyQt5.QtCore import QTimer, Qt
        self.timer = t = QTimer(self)
        t.timerType = Qt.PreciseTimer
        t.timeout.connect(self._redraw_timer_callback)
        t.start(self.redraw_interval)

    def _redraw_timer_callback(self):
        import time
        t = time.perf_counter()
        dur = t - self.last_redraw_start_time
        if t >= self.last_redraw_finish_time + self.minimum_event_processing_ratio * dur:
            # Redraw only if enough time has elapsed since last frame to process some events.
            # This keeps the user interface responsive even during slow rendering
            self.last_redraw_start_time = t
            s = self.session
            if not s.update_loop.draw_new_frame(s):
                s.ui.mouse_modes.mouse_pause_tracking()
            self.last_redraw_finish_time = time.perf_counter()

    def update_graphics_now(self):
        '''
        Redraw graphics now if there are any changes.  This is typically only used by
        mouse drag code that wants to update the graphics as responsively as possible,
        particularly when a mouse step may take significant computation, such as contour
        surface level change.  After each mouse event this is called to force a redraw.
        '''
        s = self.session
        s.update_loop.draw_new_frame(s)
            
from PyQt5.QtWidgets import QLabel
class Popup(QLabel):

    def __init__(self, graphics_window):
        from PyQt5.QtCore import Qt
        QLabel.__init__(self)
        self.setWindowFlags(self.windowFlags() | Qt.ToolTip)
        self.graphics_window = graphics_window

    def show_text(self, text, position):
        self.setText(text)
        from PyQt5.QtCore import QPoint
        self.move(self.graphics_window.mapToGlobal(QPoint(*position)))
        self.show()
