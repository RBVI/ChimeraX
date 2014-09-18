# vim: set expandtab ts=4 sw=4:

import wx
from ...graphics import View

class GraphicsWindow(View, wx.Panel):
    """
    The graphics window that displays the three-dimensional models.

    Routines that involve the window toolkit or event processing are
    handled by this class while routines that depend only on OpenGL
    are in the View base class.
    """

    def __init__(self, session, parent=None, req_size=(800,800)):
        wx.Panel.__init__(self, parent)
        if req_size:
            self.SetClientSize(*req_size)
        self.opengl_context = None
        self.opengl_canvas = OpenGLCanvas(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

        View.__init__(self, session, self.GetClientSize())

        self.set_stereo_eye_separation()

        self.timer = None
        self.redraw_interval = 16  #milliseconds

    def create_opengl_context(self):
        from wx.glcanvas import GLContext
        return GLContext(self.opengl_canvas)

    def make_opengl_context_current(self):
        if self.opengl_context is None:
            self.opengl_context = self.create_opengl_context()
            self.start_update_timer()
        self.opengl_canvas.SetCurrent(self.opengl_context)

    def redraw_timer_callback(self, evt):
        if True:
            if not self.redraw():
                self.mouse_modes.mouse_pause_tracking()

    def set_stereo_eye_separation(self, eye_spacing_millimeters=61.0):
        screen = wx.ScreenDC()
        ssize = screen.GetSizeMM()[0]
        psize = screen.GetSize()[0]
        self.camera.eye_separation_pixels = psize * eye_spacing_millimeters \
            / ssize

    def start_update_timer(self):
        if self.timer is None:
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.redraw_timer_callback, self.timer)
            self.timer.Start(self.redraw_interval)

    def swap_opengl_buffers(self):
        self.opengl_canvas.SwapBuffers()

from wx import glcanvas
class OpenGLCanvas(glcanvas.GLCanvas):

    def __init__(self, parent):
        self.graphics_window = parent
        import sys
        print(dir(glcanvas), file=sys.stderr)
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList = [
            glcanvas.WX_GL_RGBA,
            glcanvas.WX_GL_DOUBLEBUFFER,
            glcanvas.WX_GL_DEPTH_SIZE, 24,
            glcanvas.WX_GL_OPENGL_PROFILE, glcanvas.WX_GL_OPENGL_PROFILE_3_2CORE,
        ])

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)

    def OnPaint(self, evt):
        #dc = wx.PaintDC(self)
        #self.OnDraw()
        self.graphics_window.draw_graphics()

    def OnSize(self, evt):
        wx.CallAfter(self.set_viewport)
        evt.Skip()

    def set_viewport(self):
        self.graphics_window.window_size = w, h = self.GetClientSize()
        if self.graphics_window.opengl_context is not None:
            from ... import graphics
            fb = graphics.default_framebuffer()
            fb.width, fb.height = w, h
            fb.viewport = (0, 0, w, h)

    def OnMouseDown(self, evt):
        self.CaptureMouse()
        self.x, self.y, = self.last_x, self.last_y = evt.GetPosition()

    def OnMouseUp(self, evt):
        self.ReleaseMouse()

    def OnMouseMotion(self, evt):
        if evt.Dragging() and evt.LeftIsDown():
            self.last_x, self.last_y = self.x, self.y
            self.x, self.y = evt.GetPosition()
            # mousemodes...
