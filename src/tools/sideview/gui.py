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
from chimera.core.graphics.camera import CameraMode


class SideViewCameraMode(CameraMode):

    def __init__(self, main_view):
        self.main_view = main_view

    def combine_rendered_camera_views(self, render):
        from OpenGL import GL
        w, h = self.main_view.window_size
        GL.glViewport(0, 0, w, h)


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
        self.view.camera.mode = SideViewCameraMode(self.main_view)
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def OnPaint(self, event):  # noqa
        # TODO: set flag to be drawn
        # print('Sideview::OnPaint') # DEBUG
        self.draw()

    def OnSize(self, event):  # noqa
        # poor man's collapsing of OnSize events
        wx.CallAfter(self.set_viewport)
        event.Skip()

    def set_viewport(self):
        self.view.resize(*self.GetClientSize())

    def make_current(self):
        self.SetCurrent(self.view.opengl_context())

    def swap_buffers(self):
        self.SwapBuffers()

    def draw(self):
        self.view._render.shader_programs = \
            self.main_view._render.shader_programs
        opengl_context = self.view.opengl_context()
        save_make_current = opengl_context.make_current
        save_swap_buffers = opengl_context.swap_buffers
        opengl_context.make_current = self.make_current
        opengl_context.swap_buffers = self.swap_buffers
        try:
            from OpenGL.GL.GREMEDY import string_marker
            has_string_marker = string_marker.glInitStringMarkerGREMEDY()
            if has_string_marker:
                text = b"Start SideView"
                string_marker. glStringMarkerGREMEDY(len(text), text)
            main_view = self.main_view
            main_camera = main_view.camera
            view_num = None  # TODO: 0, 1 for stereo
            # TODO: make _near_far_clip public?
            near, far = main_view._near_far_clip(main_camera, view_num)
            main_pos = main_camera.get_position(view_num)
            main_axes = main_pos.axes()
            camera = self.view.camera
            camera_pos = camera.get_position()
            camera_axes = camera_pos.axes()
            camera_axes[0] = -main_axes[2]
            camera_axes[1] = -main_axes[0]
            camera_axes[2] = main_axes[1]
            center = main_pos.origin() + (far / 2) * main_camera.view_direction()
            main_view_width = main_camera.view_width(center)
            camera_pos.origin()[:] = center + camera_axes[2] * main_view_width * 5
            view_width = 1.2 * far
            camera.set_field_of_view_from_view_width(center, view_width)
            camera.redraw_needed = True
            self.view.draw(only_if_changed=False)
            if has_string_marker:
                text = b"End SideView"
                string_marker. glStringMarkerGREMEDY(len(text), text)
        finally:
            opengl_context.make_current = save_make_current
            opengl_context.swap_buffers = save_swap_buffers


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
                         session.logger, track=False)
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
