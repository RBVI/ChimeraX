from .qt import QtCore, QtGui, QtOpenGL, QtWidgets
from ...graphics import View

class Graphics_Window(QtGui.QWindow):
    '''
    A graphics window displays the 3-dimensional models.
    Routines that involve the window toolkit or event processing are handled by this class
    while routines that depend only on OpenGL are in the View base class.
    '''
    def __init__(self, session, parent=None):

        self.session = session

        QtGui.QWindow.__init__(self)
        self.widget = w = QtWidgets.QWidget.createWindowContainer(self, parent)
        self.setSurfaceType(QtGui.QSurface.OpenGLSurface)       # QWindow will be rendered with OpenGL
#        w.setFocusPolicy(QtCore.Qt.ClickFocus)
        w.setFocusPolicy(QtCore.Qt.NoFocus)

        window_size = (w.width(), w.height())		# pixels
        log = session
        drawing = session.drawing()
        self.opengl_context = QtOpenGLContext(self)
        self.view = View(drawing, window_size, self.opengl_context, log)
        session.view = self.view        # Needed by mouse modes

        self.set_stereo_eye_separation()

        self.timer = None			# Redraw timer
        self.redraw_interval = 10               # milliseconds
        self.minimum_event_processing_ratio = 0.1   # Event processing time as a fraction of time since start of last drawing
        self.last_redraw_start_time = 0
        self.last_redraw_finish_time = 0

        from . import mousemodes
        self.mouse_modes = mousemodes.QtMouseModes(self)
        self.enable_trackpad_events()

    def set_stereo_eye_separation(self, eye_spacing_millimeters = 61.0):
        # Set stereo eye spacing parameter based on physical screen size
        s = self.screen()
        ssize = s.physicalSize().width()        # millimeters
        psize = s.size().width()                # pixels
        c = self.view.camera
        c.eye_separation_pixels = psize * (eye_spacing_millimeters / ssize)

    def enable_trackpad_events(self):
        # TODO: Qt 5.1 has touch events disabled on Mac
        #        w.setAttribute(QtCore.Qt.WA_AcceptTouchEvents)
        # Qt 5.2 has touch events disabled because it slows down scrolling.  Reenable them.
        import sys
        if sys.platform == 'darwin':
            from ... import mac_os_cpp
            mac_os_cpp.accept_touch_events(int(self.winId()))

    # QWindow method
    def resizeEvent(self, e):
        s = e.size()
        w, h = s.width(), s.height()
        self.view.resize(w,h)
#
# TODO: On Mac retina display event window size is half of opengl window size.
#    Can scale width/height here, but also need mouse event positions to be scaled by 2x.
#    Not sure how to detect when app moves between non-retina and retina displays.
#    QWindow has a screenChanged signal but I did not get it in tests with Qt 5.2.
#    Also did not get moveEvent().  May need to get these on top level window?
#
#        r = self.devicePixelRatio()    # 2 on retina display, 1 on non-retina
#        w,h = int(r*w), int(r*h)
#

    # QWindow method
    def exposeEvent(self, event):
        if self.isExposed():
            if self.timer is None:
                self.start_update_timer()
            self.view.draw()

    # QWindow method
    def keyPressEvent(self, event):

        # TODO: This window should never get key events since we set widget.setFocusPolicy(NoFocus)
        # but it gets them anyways on Mac in Qt 5.2 if the graphics window is clicked.
        # So we pass them back to the main window.
        self.session.main_window.event(event)

    def set_redraw_interval(self, milliseconds):
        self.redraw_interval = milliseconds
        t = self.timer
        if not t is None:
            t.setInterval(milliseconds)

    def start_update_timer(self):
        if self.timer is None:
            self.timer = t = QtCore.QTimer(self)
            t.timerType = QtCore.Qt.PreciseTimer
            t.timeout.connect(self.redraw_timer_callback)
            t.start(self.redraw_interval)

    def redraw_timer_callback(self):
        import time
        t = time.perf_counter()
        dur = t - self.last_redraw_start_time
        if t >= self.last_redraw_finish_time + self.minimum_event_processing_ratio * dur:
            # Redraw only if enough time has elapsed since last frame to process some events.
            # This keeps the user interface responsive even during slow rendering.
            self.last_redraw_start_time = t
            self.update_graphics()
            self.last_redraw_finish_time = time.perf_counter()

    def update_graphics(self):
        if self.isExposed():
            if not self.view.draw(only_if_changed = True):
                self.mouse_modes.mouse_pause_tracking()

from ...graphics import OpenGLContext
class QtOpenGLContext(OpenGLContext):

    def __init__(self, graphics_window, shared_context = None):
        self._graphics_window = graphics_window
        self._opengl_context = None
        self._shared_context = shared_context

    def __del__(self):
        self._opengl_context.deleteLater()
        self._opengl_context = None

    def make_current(self):
        c = self._opengl_context
        if c is None:
            c = self._create_opengl_context()
        if not c.makeCurrent(self._graphics_window):
            raise RuntimeError('Could not make graphics context current')

    def swap_buffers(self):
        self._opengl_context.swapBuffers(self._graphics_window)
        
    def _create_opengl_context(self, stereo = False):

        f = self._pixel_format(stereo)
        gw = self._graphics_window
        gw.setFormat(f)
        gw.create()

        c = QtGui.QOpenGLContext(self._graphics_window)
        self._opengl_context = c
        if not self._shared_context is None:
            c.setShareContext(self._shared_context._opengl_context)
        c.setFormat(f)
        if not c.create():
            raise SystemError('Failed creating QOpenGLContext')
        c.makeCurrent(gw)

        # Write a log message indicating OpenGL version
        s = gw.session
        f = c.format()
        stereo = 'stereo' if f.stereo() else 'no stereo'
        s.show_info('OpenGL version %s, %s' % (s.view.opengl_version(), stereo))

        return c

    def _pixel_format(self, stereo = False):

        f = QtGui.QSurfaceFormat()
        f.setMajorVersion(3)
        f.setMinorVersion(3)
        f.setDepthBufferSize(24);
        f.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        f.setStereo(stereo)
        return f

    def _enable_opengl_stereo(self, enable):

        supported = self._opengl_context.format().stereo()
        if not enable or supported:
            return True

        msg = 'Stereo mode is not supported by OpenGL driver'
        s = self.graphics_window.session
        s.show_status(msg)
        s.show_info(msg)
        return False

class Secondary_Graphics_Window(QtGui.QWindow):
    '''
    A top level graphics window separate for the main window for example to render to Oculus Rift headset.
    It has its own opengl context that shares state with the main graphics window context.
    '''
    def __init__(self, title, session, show = True):

        self.session = session
        QtGui.QWindow.__init__(self)
        # Use main window as a parent so this window is closed if main window gets closed.
        parent = session.main_window
        self.widget = w = QtWidgets.QWidget.createWindowContainer(self, parent, QtCore.Qt.Window)
        self.setSurfaceType(QtGui.QSurface.OpenGLSurface)       # QWindow will be rendered with OpenGL
        w.setWindowTitle(title)
        if show:
            w.show()

        shared_context = session.main_window.graphics_window.opengl_context
        self.opengl_context = QtOpenGLContext(self, shared_context)
        self.primary_opengl_context = shared_context

    def close(self):
        self.opengl_context = None
        self.widget.close()
        self.widget = None

    def full_screen(self, width, height):
        d = self.session.application.desktop()
        ow = self.widget
        for s in range(d.screenCount()):
            g = d.screenGeometry(s)
            if g.width() == width and g.height() == height:
                ow.move(g.left(), g.top())
                break
        ow.resize(width,height)
        ow.showFullScreen()

    def move_window_to_primary_screen(self):
        d = self.session.application.desktop()
        s = d.primaryScreen()
        g = d.screenGeometry(s)
        ow = self.widget
        ow.showNormal()     # Exit full screen mode.  
        x,y = (g.width() - ow.width())//2, (g.height() - ow.height())//2
        def move_window(ow=ow, x=x, y=y):
            ow.move(x, y)
        # TODO: On Mac OS 10.9 going out of full-screen takes a second during which
        #   moving the window to the primary display does nothing.
        from ...ui.qt.qt import QtCore
        QtCore.QTimer.singleShot(1500, move_window)
