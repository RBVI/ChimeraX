# vim: set expandtab ts=4 sw=4:

import wx
from ...graphics import View

class GraphicsWindow(wx.Panel):
    """
    The graphics window that displays the three-dimensional models.

    Routines that involve the window toolkit or event processing are
    handled by this class while routines that depend only on OpenGL
    are in the View base class.
    """

    def __init__(self, session, parent=None):
        self.session = session
        wx.Panel.__init__(self, parent,
            style=wx.TAB_TRAVERSAL|wx.NO_BORDER|wx.WANTS_CHARS)

        self.opengl_canvas = OpenGLCanvas(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

        log = session
        self.view = View(session.drawing(), self.GetClientSize(),
                         WxOpenGLContext(self.opengl_canvas), log)

        self.set_stereo_eye_separation()

        self.timer = None
        self.redraw_interval = 16  #milliseconds
        #TODO: maybe redraw interval should be 10 msec to reduce
        # frame drops at 60 frames/sec

        from . import mousemodes
        self.mouse_modes = mousemodes.WxMouseModes(self)

        self.start_update_timer()

    def redraw_timer_callback(self, evt):
        if True:
            if not self.view.draw_if_changed():
                self.mouse_modes.mouse_pause_tracking()

    def set_stereo_eye_separation(self, eye_spacing_millimeters=61.0):
        screen = wx.ScreenDC()
        ssize = screen.GetSizeMM()[0]
        psize = screen.GetSize()[0]
        self.view.camera.eye_separation_pixels = psize * eye_spacing_millimeters \
            / ssize

    def start_update_timer(self):
        if self.timer is None:
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.redraw_timer_callback, self.timer)
            self.timer.Start(self.redraw_interval)

from ...graphics import OpenGLContext
class WxOpenGLContext(OpenGLContext):
    def __init__(self, opengl_canvas):
        self._opengl_canvas = opengl_canvas
        self._opengl_context = None

    def _create_context(self):
        from wx.glcanvas import GLContext
        return GLContext(self._opengl_canvas)

    def make_current(self):
        if self._opengl_context is None:
            self._opengl_context = self._create_context()
        self._opengl_canvas.SetCurrent(self._opengl_context)

    def swap_buffers(self):
        self._opengl_canvas.SwapBuffers()

from wx import glcanvas
class OpenGLCanvas(glcanvas.GLCanvas):

    def __init__(self, parent):
        self.graphics_window = parent
        attribs = [ glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER,
            glcanvas.WX_GL_OPENGL_PROFILE, glcanvas.WX_GL_OPENGL_PROFILE_3_2CORE
            ]
        gl_supported = glcanvas.GLCanvas.IsDisplaySupported
        if not gl_supported(attribs):
            raise AssertionError("Required OpenGL capabilities RGBA and/or"
                " double buffering and/or OpenGL 3 not supported")
        for depth in range(32, 0, -8):
            test_attribs = attribs + [glcanvas.WX_GL_DEPTH_SIZE, depth]
            if gl_supported(test_attribs):
                attribs = test_attribs
                print("Using {}-bit OpenGL depth buffer".format(depth))
                break
        else:
            raise AssertionError("Required OpenGL depth buffer capability"
                " not supported")
        test_attribs = attribs + [glcanvas.WX_GL_STEREO]
        if gl_supported(test_attribs):
            attribs = test_attribs
        else:
            print("Stereo mode is not supported by OpenGL driver")
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs,
            style=wx.WANTS_CHARS)

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_CHAR, self.OnChar)

    def OnChar(self, event):
        cmd_line = self.graphics_window.session.main_window._command_line
        if event.KeyCode == 13:
            cmd_line.OnEnter(event)
        else:
            cmd_line.text.EmulateKeyPress(event)
        event.Skip()

    def OnPaint(self, evt):
        #dc = wx.PaintDC(self)
        #self.OnDraw()
        self.graphics_window.view.draw()

    def OnSize(self, evt):
        wx.CallAfter(self.set_viewport)
        evt.Skip()

    def set_viewport(self):
        w, h = self.GetClientSize()
        self.graphics_window.view.resize(w,h)
