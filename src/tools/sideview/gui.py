# vi: set expandtab ts=4 sw=4:

# ToolUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
# Since ToolInstance derives from core.session.State, which
# is an abstract base class, ToolUI classes must implement
#   "take_snapshot" - return current state for saving
#   "restore_snapshot" - restore from given state
#   "reset_state" - reset to data-less state
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
import wx
from wx import glcanvas
from chimera.core.tools import ToolInstance


class SideViewCanvas(glcanvas.GLCanvas):

    def __init__(self, parent, view, main_view):
        import sys
        attribs = [
            glcanvas.WX_GL_RGBA,
            glcanvas.WX_GL_DOUBLEBUFFER,
            glcanvas.WX_GL_DEPTH_SIZE, 1,
            0  # terminates list for OpenGL
        ]
        if sys.platform.startswith('darwin'):
            attribs[-1:-1] = [
                glcanvas.WX_GL_OPENGL_PROFILE,
                glcanvas.WX_GL_OPENGL_PROFILE_3_2CORE
            ]
        if not glcanvas.GLCanvas.IsDisplaySupported(attribs):
            raise AssertionError(
                    "Missing required OpenGL capabilities for Side View")
        self.view = view
        self.main_view = main_view
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def OnPaint(self, event):  # noqa
        # TODO: set flag to be drawn
        print('Sideview::OnPaint')
        self.SetCurrent(self.view.opengl_context())
        from OpenGL import GL
        width, height = self.view.window_size
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # TODO: draw
        self.SwapBuffers()
        # reset viewport to main_view's
        width, height = self.main_view.window_size
        GL.glViewport(0, 0, width, height)

    def OnSize(self, event):  # noqa
        # poor man's collapsing of OnSize events
        wx.CallAfter(self.set_viewport)
        event.Skip()

    def set_viewport(self):
        self.view.resize(*self.GetClientSize())


class ToolUI(ToolInstance):

    SIZE = (300, 200)
    VERSION = 1

    def __init__(self, session, **kw):
        super().__init__(session, **kw)
        import weakref
        self._session = weakref.ref(session)
        from chimera.core.ui.tool_api import ToolWindow
        self.tool_window = ToolWindow("Side view",
                                      session, size=self.SIZE)
        parent = self.tool_window.ui_area
        # UI content code
        from chimera.core.graphics.view import View
        self.opengl_context = oc = session.main_view.opengl_context()
        self.view = View(session.models.drawing, wx.DefaultSize, oc,
                         session.logger)
        self.opengl_canvas = SideViewCanvas(parent, self.view,
                                            session.main_view)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")

        # Add to running tool list for session (not required)
        session.tools.add([self])

    def OnEnter(self, event):  # noqa
        # session = self._session()  # resolve back reference
        # Handle event
        pass

    def make_context_current(self):
        self.opengl_canvas.SetCurrent(self.opengl_context)

    def swap_buffers(self):
        self.opengl_canvas.SwapBuffers()

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        version = self.VERSION
        data = {}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.VERSION or len(data) > 0:
            raise RuntimeError("unexpected version or data")
        from chimera.core.session import State
        if phase == State.PHASE1:
            # Restore all basic-type attributes
            pass
        else:
            # Resolve references to objects
            pass

    def reset_state(self):
        pass

    #
    # Override ToolInstance delete method to clean up
    #
    def delete(self):
        session = self._session()  # resolve back reference
        self.tool_window.shown = False
        self.tool_window.destroy()
        session.tools.remove([self])
        super().delete()

    def display(self, b):
        self.tool_window.shown = b

    def display_name(self):
        return "custom name for running tool"
