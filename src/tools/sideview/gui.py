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
from chimera.core.geometry.place import Place


class _PixelLocations:
    pass


class SideViewCanvas(glcanvas.GLCanvas):

    EyeSize = 4     # half size really
    TOP_SIDE = 1
    RIGHT_SIDE = 2

    def __init__(self, parent, view, main_view, size):
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
        self.view.camera.ortho = True
        self.side = self.TOP_SIDE
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs,
                                   size=size)

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP,  self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)

        self.locations = loc = _PixelLocations()
        loc.eye = 0, 0, 0   # x, y coords of eye
        loc.near = 0        # X coord of near plane
        loc.far = 0         # Y coord of near plane
        loc.bottom = 0      # bottom of clipping planes
        loc.top = 0         # top of clipping planes
        loc.far_bottom = 0  # right clip intersect far
        loc.far_top = 0     # left clip intersect far
        loc.view_width = 0  # help conversion to coordinates

        from chimera.core.graphics import Drawing
        self.applique = Drawing('sideview')
        self.applique.display_style = Drawing.Mesh
        self.applique.use_lighting = False
        self.view.add_2d_overlay(self.applique)
        self.main_view.add_rendered_frame_callback(self._redraw)

    def on_destroy(self, event):
        self.main_view.remove_rendered_frame_callback(self._redraw)

    def _redraw(self):
        # wx.CallAfter(self.draw)
        self.draw()

    def on_paint(self, event):
        # TODO: set flag to be drawn
        # print('Sideview::OnPaint') # DEBUG
        if self.view.window_size[0] == -1:
            return
        self.draw()

    def on_size(self, event):
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
        from math import pi, tan
        from numpy import array, float32, uint8, int32
        self.view._render.shader_programs = \
            self.main_view._render.shader_programs
        self.view._render.current_shader_program = None
        # self.view.set_background_color((.3, .3, .3, 1))  # DEBUG
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
            camera_pos = Place()
            camera_axes = camera_pos.axes()
            if self.side == self.TOP_SIDE:
                fov = main_camera.field_of_view
                camera_axes[0] = -main_axes[2]
                camera_axes[1] = -main_axes[0]
                camera_axes[2] = main_axes[1]
                center = main_pos.origin() + (far / 2) * \
                    main_camera.view_direction()
                main_view_width = main_camera.view_width(center)
                camera_pos.origin()[:] = center + camera_axes[2] * \
                    main_view_width * 5
            else:
                w, h = self.main_view.window_size
                fov = main_camera.field_of_view * h / w  # TODO: confirm
                # TODO
            camera.position = camera_pos

            # figure out how big to make applique
            # eye and lines to far plane must be on screen
            width, height = self.view.window_size
            loc = self.locations
            loc.bottom = .05 * height
            loc.top = .95 * height
            ratio = tan(0.5 * fov * pi / 180)
            if height / width >= (1 - (8 * self.EyeSize) / height):
                loc.view_width = 1.1 * far
                loc.eye = array([.05 / 1.1 * width, height / 2, 0],
                                dtype=float32)
                loc.near = (.05 + near / far) / 1.1 * width
                loc.far = 1.05 / 1.1 * width
                loc.far_top = .5 * height + ratio * width / 1.1
                loc.far_bottom = .5 * height - ratio * width / 1.1
            else:
                loc.far_bottom = loc.bottom
                loc.far_top = loc.top
                f = .45 * height / ratio
                n = f * near / far
                loc.eye = array([.5 * width - f / 2, height / 2, 0],
                                dtype=float32)
                loc.near = loc.eye[0] + n
                loc.far = .5 * width + f / 2
                loc.view_width = far * 1.1 * width / height

            if self.side == self.TOP_SIDE:
                camera.set_field_of_view_from_view_width(center, loc.view_width)
            else:
                w, h = self.main_view.window_size
                view_width *= w / h
                camera.set_field_of_view_from_view_width(center, loc.view_width)
            self.applique.color = array([255, 0, 0, 255], dtype=uint8)
            es = self.EyeSize
            self.applique.vertices = array([
                loc.eye + [-es, -es, 0], loc.eye + [-es, es, 0],
                loc.eye + [es, es, 0], loc.eye + [es, -es, 0],
                (loc.near, loc.bottom, 0), (loc.near, loc.top, 0),
                (loc.far, loc.bottom, 0), (loc.far, loc.top, 0),
                (0, 0, 0), (0, 0, 0),
                (0, 0, 0), (0, 0, 0),
            ], dtype=float32)
            if main_camera.ortho:
                self.applique.vertices[8] = (loc.near, loc.far_top, 0)
                self.applique.vertices[9] = (loc.far, loc.far_top, 0)
                self.applique.vertices[10] = (loc.near, loc.far_bottom, 0)
                self.applique.vertices[11] = (loc.far, loc.far_bottom, 0)
            else:
                self.applique.vertices[8] = loc.eye
                self.applique.vertices[9] = (loc.far, loc.far_top, 0)
                self.applique.vertices[10] = loc.eye
                self.applique.vertices[11] = (loc.far, loc.far_bottom, 0)
            self.applique.triangles = array([
                [0, 1], [1, 2], [2, 3], [3, 0],  # eye box
                [4, 5],    # near plane
                [6, 7],    # far plane
                [8, 9],    # left plane
                [10, 11],  # right plane
            ], dtype=int32)
            from OpenGL import GL
            GL.glViewport(0, 0, width, height)
            self.view.draw()
            if has_string_marker:
                text = b"End SideView"
                string_marker. glStringMarkerGREMEDY(len(text), text)
        finally:
            opengl_context.make_current = save_make_current
            opengl_context.swap_buffers = save_swap_buffers
            self.main_view._render.current_shader_program = None
            from OpenGL import GL
            w, h = self.main_view.window_size
            GL.glViewport(0, 0, w, h)

    def on_left_down(self, event):
        x, y = event.GetPosition()
        eye_x, eye_y = self.locations.eye[0:2]
        es = self.EyeSize
        if eye_x - es <= x <= eye_x + es and eye_y - es <= y <= eye_y + es:
            self.x, self.y = x, y
            self.CaptureMouse()

    def on_left_up(self, event):
        if self.HasCapture():
            self.ReleaseMouse()

    def on_motion(self, event):
        if not self.HasCapture() or not event.Dragging():
            return
        x, y = event.GetPosition()
        v = self.main_view
        delta = (x - self.x) / 5
        psize = self.locations.far / self.locations.view_width
        shift = v.camera.position.apply_without_translation(
            (0, 0, delta * psize))
        v.translate(shift)
        self.x, self.y = x, y


class ToolUI(ToolInstance):

    SIZE = (300, 200)
    VERSION = 1

    def __init__(self, session, **kw):
        super().__init__(session, **kw)
        from chimera.core.ui.tool_api import ToolWindow
        self.tool_window = ToolWindow("Side View", session, size=self.SIZE)
        parent = self.tool_window.ui_area

        # UI content code
        from chimera.core.graphics.view import View
        self.opengl_context = oc = session.main_view.opengl_context()
        self.view = View(session.models.drawing, wx.DefaultSize, oc,
                         session.logger, track=False)
        self.opengl_canvas = SideViewCanvas(parent, self.view,
                                            session.main_view, self.SIZE)

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
        self.tool_window.shown = False
        self.tool_window.destroy()
        self.session.tools.remove([self])
        super().delete()

    def display(self, b):
        self.tool_window.shown = b

    def display_name(self):
        return "Side View"
