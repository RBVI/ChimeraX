from .qt import QtCore, QtGui, QtOpenGL, QtWidgets
from ..graphics import View

class Graphics_Window(View, QtGui.QWindow):
    '''
    A graphics window displays the 3-dimensional models.
    Routines that involve the window toolkit or event processing are handled by this class
    while routines that depend only on OpenGL are in the View base class.
    '''
    def __init__(self, session, parent=None):

        QtGui.QWindow.__init__(self)
        self.widget = w = QtWidgets.QWidget.createWindowContainer(self, parent)
        self.setSurfaceType(QtGui.QSurface.OpenGLSurface)       # QWindow will be rendered with OpenGL
#        w.setFocusPolicy(QtCore.Qt.ClickFocus)
        w.setFocusPolicy(QtCore.Qt.NoFocus)

        window_size = (w.width(), w.height())		# pixels
        View.__init__(self, session, window_size)

        self.set_stereo_eye_separation()

        self.timer = None			# Redraw timer
        self.redraw_interval = 16               # milliseconds
        # TODO: Maybe redraw interval should be 10 msec to reduce frame drops at 60 frames/sec

        from . import mousemodes
        self.mouse_modes = mousemodes.Mouse_Modes(self)
        self.enable_trackpad_events()

    def set_stereo_eye_separation(self, eye_spacing_millimeters = 61.0):
        # Set stereo eye spacing parameter based on physical screen size
        s = self.screen()
        ssize = s.physicalSize().width()        # millimeters
        psize = s.size().width()                # pixels
        self.camera.eye_separation_pixels = psize * (eye_spacing_millimeters / ssize)

    def enable_trackpad_events(self):
        # TODO: Qt 5.1 has touch events disabled on Mac
        #        w.setAttribute(QtCore.Qt.WA_AcceptTouchEvents)
        # Qt 5.2 has touch events disabled because it slows down scrolling.  Reenable them.
        import sys
        if sys.platform == 'darwin':
            from .. import mac_os_cpp
            mac_os_cpp.accept_touch_events(int(self.winId()))

    # QWindow method
    def resizeEvent(self, e):
        s = e.size()
        w, h = s.width(), s.height()
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
        self.window_size = w, h
        if not self.opengl_context is None:
            from .. import graphics
            fb = graphics.default_framebuffer()
            fb.width, fb.height = w,h
            fb.viewport = (0,0,w,h)
#            self.render.set_viewport(0,0,w,h)

    # QWindow method
    def exposeEvent(self, event):
        if self.isExposed():
            self.draw_graphics()

    def keyPressEvent(self, event):

        # TODO: This window should never get key events since we set widget.setFocusPolicy(NoFocus)
        # but it gets them anyways on Mac in Qt 5.2 if the graphics window is clicked.
        # So we pass them back to the main window.
        self.session.main_window.event(event)

    def create_opengl_context(self):

        f = self.pixel_format(stereo = True)
        self.setFormat(f)
        self.create()

        c = QtGui.QOpenGLContext(self)
        c.setFormat(f)
        if not c.create():
            raise SystemError('Failed creating QOpenGLContext')
        c.makeCurrent(self)

        return c

    def pixel_format(self, stereo = False):

        f = QtGui.QSurfaceFormat()
        f.setMajorVersion(3)
        f.setMinorVersion(2)
        f.setDepthBufferSize(24);
        f.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        f.setStereo(stereo)
        return f

    def enable_opengl_stereo(self, enable):

        supported = self.opengl_context.format().stereo()
        if not enable or supported:
            return True

        msg = 'Stereo mode is not supported by OpenGL driver'
        s = self.session
        s.show_status(msg)
        s.show_info(msg)
        return False

        # TODO: Current strategy for handling stereo is to request a stereo OpenGL context
        # when graphics window created.  Use it for both stereo and mono display without
        # switching contexts. There are several obstacles to switching contexts.  First,
        # we need to share context state.  When tested with Qt 5.1 this caused crashes in
        # the QCocoaCreateOpenGLContext() routine, probably because the pixel format was null
        # perhaps because sharing was not supported.  A second problem is that we need to
        # switch the format of the QWindow.  It is not clear from the Qt documentation if this
        # is possible.  My tests failed.  The QWindow.setFormat() docs say "calling that function
        # after create() has been called will not re-resolve the surface format of the native surface."
        # Maybe calling destroy on the QWindow, then setFormat() and create() would work.  Did not try.
        # It may be necessary to simply destroy the old QWindow and QWidget container and make a new
        # one. A third difficulty is that OpenGL does not allow sharing VAOs between contexts.
        # Drawings use VAOs, so those would have to be destroyed and recreated.  Sharing does
        # handle VBOs, textures and shaders.
        #
        # Test code follows.
        #
        f = self.pixel_format(enable)
        c = QtGui.QOpenGLContext(self)
        c.setFormat(f)
        c.setShareContext(self.opengl_context)  # Share shaders, vbos and textures, but not VAOs.
        if not c.create() or (enable and not c.format().stereo()):
            if enable:
                msg = 'Stereo mode is not supported by OpenGL driver'
            else:
                msg = 'Failed changing graphics mode'
            s = self.session
            s.show_status(msg)
            s.show_info(msg)
            return False
        self.opengl_context = c
        c.makeCurrent(self)

        self.setFormat(f)
        if not self.create():
            raise SystemError('Failed to create QWindow with new format')

        return True

    def start_update_timer(self):

        self.timer = t = QtCore.QTimer(self)
        t.timeout.connect(self.redraw_timer_callback)
        t.start(self.redraw_interval)

    def redraw_timer_callback(self):
        if not self.redraw():
            self.mouse_modes.mouse_pause_tracking()
