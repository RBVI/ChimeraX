# vim: set expandtab ts=4 sw=4:

import wx

class GraphicsWindow(wx.Panel):
    """
    The graphics window that displays the three-dimensional models.
    """

    def __init__(self, parent, ui):
        wx.Panel.__init__(self, parent,
            style=wx.TAB_TRAVERSAL|wx.NO_BORDER|wx.WANTS_CHARS)
        self.opengl_context = None
        self.opengl_canvas = OpenGLCanvas(self, ui)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

from wx import glcanvas
class OpenGLCanvas(glcanvas.GLCanvas):

    def __init__(self, parent, ui):
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

        self.Bind(wx.EVT_CHAR, ui.forward_keystroke)
