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
        from wx.glcanvas import GLCanvas
        self.gl_canvas = GLCanvas(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.gl_canvas, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)

        View.__init__(self, session, self.GetClientSize())

