# vim: set expandtab ts=4 sw=4:

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
from chimera.core.tools import ToolInstance


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
        from chimera.core.ui.graphics import OpenGLCanvas
        self.opengl_canvas = OpenGLCanvas(parent)
        from wx.glcanvas import GLContext
        oc = self.opengl_context = GLContext(self.opengl_canvas)
        oc.make_current = self.make_context_current
        oc.swap_buffers = self.swap_buffers
        # OpenGLCanvas expects 'view' in parent
        parent.view = View(session.models.drawing, wx.DefaultSize, oc,
                           session.logger)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")

        # Add to running tool list for session (not required)
        session.tools.add([self])

    def OnEnter(self, event):
        session = self._session()  # resolve back reference
        # Handle event

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
