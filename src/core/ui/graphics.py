# vim: set expandtab ts=4 sw=4:

import wx

class GraphicsWindow(wx.Panel):
    """
    The graphics window that displays the three-dimensional models.
    """

    def __init__(self, parent, ui):
        wx.Panel.__init__(self, parent,
            style=wx.TAB_TRAVERSAL|wx.NO_BORDER|wx.WANTS_CHARS)
        self.timer = None
        self.opengl_canvas = OpenGLCanvas(self, ui)
        from wx.glcanvas import GLContext
        oc = self.opengl_context = GLContext(self.opengl_canvas)
        oc.make_current = self.make_context_current
        oc.swap_buffers = self.swap_buffers
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

        from ..graphics.view import View
        from ..graphics.drawing import Drawing
        drawing = Drawing("root")
        self.view = View(drawing, self.GetClientSize(), oc,
            ui.session.logger)
        ui.session.replace_attribute('main_drawing', drawing)
        ui.session.replace_attribute('main_view', self.view)

        self.redraw_interval = 16 # milliseconds
        # perhaps redraw interval should be 10 to reduce
        # frame drops at 60 frames/sec

    def make_context_current(self):
        # creates context if needed
        if self.timer is None:
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self._redraw_timer_callback, self.timer)
            self.timer.Start(self.redraw_interval)
        self.opengl_canvas.SetCurrent(self.opengl_context)

    def swap_buffers(self):
        self.opengl_canvas.SwapBuffers()

    def _redraw_timer_callback(self, event):
        self.view.draw(only_if_changed=True)

from wx import glcanvas
class OpenGLCanvas(glcanvas.GLCanvas):

    def __init__(self, parent, ui):
        self.graphics_window = parent
        attribs = [ glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER ]
        import sys
        if sys.platform.startswith('darwin'):
            attribs += [
                glcanvas.WX_GL_OPENGL_PROFILE,
                glcanvas.WX_GL_OPENGL_PROFILE_3_2CORE
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
            # TODO: keep track of fact that 3D stereo is available, but
            # don't use it
            pass
        else:
            print("Stereo mode is not supported by OpenGL driver")
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs,
            style=wx.WANTS_CHARS)

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.Bind(wx.EVT_CHAR, ui.forward_keystroke)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def OnPaint(self, event):
        self.graphics_window.view.draw()

    def OnSize(self, event):
        wx.CallAfter(self.set_viewport)
        event.Skip()

    def set_viewport(self):
        self.graphics_window.view.resize(*self.GetClientSize())
