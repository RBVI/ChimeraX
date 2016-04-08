# vim: set expandtab ts=4 sw=4:

# ToolUI should inherit from ToolInstance if they will be
# registered with the tool state manager.
#
# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
import wx
from wx import glcanvas
from chimerax.core.tools import ToolInstance
from chimerax.core.geometry import Place
from chimerax.core.graphics import View, Camera


class _PixelLocations:
    pass


class OrthoCamera(Camera):
    """A limited camera for the Side View without field_of_view"""

    def __init__(self):
        Camera.__init__(self)
        self.position = Place()

        self.field_width = 1

    def get_position(self, view_num=None):
        return self.position

    def number_of_views(self):
        return 1

    def set_render_target(self, view_num, render):
        render.set_mono_buffer()

    def combine_rendered_camera_views(self, render):
        return

    def projection_matrix(self, near_far_clip, view_num, window_size):
        near, far = near_far_clip
        ww, wh = window_size
        aspect = wh / ww
        w = self.field_width
        h = w * aspect
        left, right, bot, top = -0.5 * w, 0.5 * w, -0.5 * h, 0.5 * h
        from chimerax.core.graphics.camera import ortho
        pm = ortho(left, right, bot, top, near, far)
        return pm

    def view_width(self, center):
        return self.field_width


class SideViewCanvas(glcanvas.GLCanvas):

    EyeSize = 4     # half size really
    TOP_SIDE = 1
    RIGHT_SIDE = 2

    ON_NOTHING = 0
    ON_EYE = 1
    ON_NEAR = 2
    ON_FAR = 3

    def __init__(self, parent, view, session, panel, size, side=RIGHT_SIDE):
        attribs = session.ui.opengl_attribs
        if not glcanvas.GLCanvas.IsDisplaySupported(attribs):
            raise AssertionError(
                "Missing required OpenGL capabilities for Side View")
        self.view = view
        self.session = session
        self.panel = panel
        self.main_view = session.main_view
        self.side = side
        # self.side = self.TOP_SIDE  # DEBUG
        self.moving = self.ON_NOTHING
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs,
                                   size=size)

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)

        self.locations = loc = _PixelLocations()
        loc.eye = 0, 0, 0   # x, y coordinates of eye
        loc.near = 0        # X coordinate of near plane
        loc.far = 0         # Y coordinate of near plane
        loc.bottom = 0      # bottom of clipping planes
        loc.top = 0         # top of clipping planes
        loc.far_bottom = 0  # right clip intersect far
        loc.far_top = 0     # left clip intersect far

        from chimerax.core.graphics import Drawing
        self.applique = Drawing('sideview')
        self.applique.display_style = Drawing.Mesh
        self.applique.use_lighting = False
        self.view.add_2d_overlay(self.applique)
        self.handler = session.triggers.add_handler('frame drawn', self._redraw)

    def on_destroy(self, event):
        self.session.triggers.delete_handler(self.handler)

    def _redraw(self, *_):
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
        try:
            self.view.resize(*self.GetClientSize())
        except RuntimeError:
            # wx.CallAfter executes after being destroyed
            pass

    def make_current(self):
        self.SetCurrent(self.view.opengl_context())

    def swap_buffers(self):
        self.SwapBuffers()

    def draw(self):
        ww, wh = self.main_view.window_size
        if ww <= 0 or wh <= 0:
            return
        width, height = self.view.window_size
        if width <= 0 or height <= 0:
            return
        from math import tan, atan, radians
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
                string_marker.glStringMarkerGREMEDY(len(text), text)
            main_view = self.main_view
            main_camera = main_view.camera
            ortho = hasattr(main_camera, 'field_width')
            view_num = None  # TODO: 0, 1 for stereo

            camera = self.view.camera
            # fov is sideview's vertical field of view,
            # unlike a camera, where it is the horizontal field of view
            # TODO: Handle orthographic main_camera which has no "field_of_view" attribute.
            if self.side == self.TOP_SIDE:
                fov = radians(main_camera.field_of_view) if hasattr(main_camera, 'field_of_view') else 45
            else:
                fov = (2 * atan(wh / ww * tan(radians(main_camera.field_of_view / 2)))
                       if hasattr(main_camera, 'field_of_view') else 45)
            main_pos = main_camera.get_position(view_num)
            near, far = main_view.near_far_distances(main_camera, view_num)
            planes = self.main_view.clip_planes
            near_plane = planes.find_plane('near')
            button = self.panel.auto_clip_near
            if near_plane:
                near = near_plane.offset(main_pos.origin())
                if button.GetValue():
                    button.SetValue(False)
            else:
                if not button.GetValue():
                    button.SetValue(True)
            far_plane = planes.find_plane('far')
            button = self.panel.auto_clip_far
            if far_plane:
                far = -far_plane.offset(main_pos.origin())
                if button.GetValue():
                    button.SetValue(False)
            else:
                if not button.GetValue():
                    button.SetValue(True)
            if not self.moving:
                main_axes = main_pos.axes()
                camera_pos = Place()
                camera_axes = camera_pos.axes()
                if self.side == self.TOP_SIDE:
                    camera_axes[0] = -main_axes[2]
                    camera_axes[1] = -main_axes[0]
                    camera_axes[2] = main_axes[1]
                else:
                    camera_axes[0] = -main_axes[2]
                    camera_axes[1] = main_axes[1]
                    camera_axes[2] = main_axes[0]
                center = main_pos.origin() + (.5 * far) * \
                    main_camera.view_direction()
                main_view_width = main_camera.view_width(center)
                if main_view_width is None:
                    main_view_width = far
                camera_pos.origin()[:] = center + camera_axes[2] * \
                    main_view_width * 5
                camera.position = camera_pos

            # figure out how big to make applique
            # eye and lines to far plane must be on screen
            loc = self.locations
            loc.bottom = .05 * height
            loc.top = .95 * height
            ratio = tan(0.5 * fov)
            if self.moving:
                eye = self.view.win_coord(main_pos.origin())
                eye[2] = 0
                loc.eye = eye
                if near_plane:
                    win_pt = self.view.win_coord(near_plane.plane_point)
                    loc.near = win_pt[0]
                if far_plane:
                    win_pt = self.view.win_coord(far_plane.plane_point)
                    loc.far = win_pt[0]
            elif ratio * width / 1.1 < .45 * height:
                camera.field_width = 1.1 * far
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
                camera.field_width = far * width / f

            self.applique.vertex_colors = array([[255, 0, 0, 255]] * 12,
                                                dtype=uint8)
            if self.moving == self.ON_EYE:
                colors = self.applique.vertex_colors
                colors[0] = colors[1] = colors[2] = colors[3] = [255, 255, 0, 255]
            elif self.moving == self.ON_NEAR:
                colors = self.applique.vertex_colors
                colors[4] = colors[5] = [255, 255, 0, 255]
            elif self.moving == self.ON_FAR:
                colors = self.applique.vertex_colors
                colors[6] = colors[7] = [255, 255, 0, 255]
            es = self.EyeSize
            old_vertices = self.applique.vertices
            self.applique.vertices = array([
                loc.eye + [-es, -es, 0], loc.eye + [-es, es, 0],
                loc.eye + [es, es, 0], loc.eye + [es, -es, 0],
                (loc.near, loc.bottom, 0), (loc.near, loc.top, 0),
                (loc.far, loc.bottom, 0), (loc.far, loc.top, 0),
                (0, 0, 0), (0, 0, 0),
                (0, 0, 0), (0, 0, 0),
            ], dtype=float32)
            if ortho:
                self.applique.vertices[8] = (loc.near, loc.far_top, 0)
                self.applique.vertices[9] = (loc.near, loc.far_bottom, 0)
            else:
                self.applique.vertices[8] = loc.eye
                self.applique.vertices[9] = loc.eye
            if self.moving and old_vertices is not None:
                self.applique.vertices[10] = old_vertices[10]
                self.applique.vertices[11] = old_vertices[11]
            else:
                self.applique.vertices[10] = (loc.far, loc.far_top, 0)
                self.applique.vertices[11] = (loc.far, loc.far_bottom, 0)
            self.applique.triangles = array([
                [0, 1], [1, 2], [2, 3], [3, 0],  # eye box
                [4, 5],    # near plane
                [6, 7],    # far plane
                [8, 10],   # left plane
                [9, 11],   # right plane
            ], dtype=int32)
            from OpenGL import GL
            GL.glViewport(0, 0, width, height)
            self.view.draw()
            if has_string_marker:
                text = b"End SideView"
                string_marker.glStringMarkerGREMEDY(len(text), text)
        finally:
            opengl_context.make_current = save_make_current
            opengl_context.swap_buffers = save_swap_buffers
            self.main_view._render.current_shader_program = None
            from OpenGL import GL
            GL.glViewport(0, 0, ww, wh)

    def on_left_down(self, event):
        x, y = event.GetPosition()
        eye_x, eye_y = self.locations.eye[0:2]
        near = self.locations.near
        far = self.locations.far
        es = self.EyeSize
        if eye_x - es <= x <= eye_x + es and eye_y - es <= y <= eye_y + es:
            self.moving = self.ON_EYE
        elif near - es <= x <= near + es:
            self.moving = self.ON_NEAR
        elif far - es <= x <= far + es:
            self.moving = self.ON_FAR
        else:
            return
        self.x, self.y = x, y
        self.CaptureMouse()
        self.Refresh()

    def on_left_up(self, event):
        if self.moving:
            self.moving = self.ON_NOTHING
            self.ReleaseMouse()
            self.Refresh()

    def on_motion(self, event):
        if not self.HasCapture() or not event.Dragging():
            return
        x, y = event.GetPosition()
        diff_x = x - self.x
        self.x, self.y = x, y
        psize = self.view.pixel_size()
        shift = self.main_view.camera.position.apply_without_translation(
            (0, 0, diff_x * psize))
        if self.moving == self.ON_EYE:
            main_camera = self.main_view.camera
            ortho = hasattr(main_camera, 'field_width')
            if ortho:
                size = min(self.view.window_size)
                # factor = 1 + diff_x / size
                factor = 10 ** (diff_x / size)
                main_camera.field_width /= factor
                main_camera.redraw_needed = True
                #self.main_view.redraw_needed = True
            else:
                self.main_view.translate(shift)
        elif self.moving == self.ON_NEAR:
            v = self.main_view
            planes = v.clip_planes
            p = planes.find_plane('near')
            if p:
                plane_point = p.plane_point
            else:
                near, far = v.near_far_distances(v.camera, None)
                camera_pos = v.camera.position.origin()
                vd = v.camera.view_direction()
                plane_point = camera_pos + near * vd
            planes.set_clip_position('near', plane_point - shift, v.camera)
        elif self.moving == self.ON_FAR:
            v = self.main_view
            planes = v.clip_planes
            p = planes.find_plane('far')
            if p:
                plane_point = p.plane_point
            else:
                near, far = v.near_far_distances(v.camera, None)
                camera_pos = v.camera.position.origin()
                vd = v.camera.view_direction()
                plane_point = camera_pos + far * vd
            planes.set_clip_position('far', plane_point - shift, v.camera)


class SideViewUI(ToolInstance):

    SIZE = (300, 200)

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area

        # UI content code
        self.opengl_context = oc = session.main_view.opengl_context()
        self.view = View(session.models.drawing, window_size=wx.DefaultSize, opengl_context=oc)
        # TODO: from chimerax.core.graphics.camera import OrthographicCamera
        self.view.camera = OrthoCamera()
        if self.display_name.startswith('Top'):
            side = SideViewCanvas.TOP_SIDE
        else:
            side = SideViewCanvas.RIGHT_SIDE
        self.opengl_canvas = SideViewCanvas(
            parent, self.view, session, self, self.SIZE, side=side)
        auto_clip = wx.StaticText(parent, label="auto clip:")
        self.auto_clip_near = wx.CheckBox(parent, label="near")
        self.auto_clip_near.SetValue(True)
        parent.Bind(wx.EVT_CHECKBOX, self.on_autoclip_near, self.auto_clip_near)
        self.auto_clip_far = wx.CheckBox(parent, label="far")
        self.auto_clip_far.SetValue(True)
        parent.Bind(wx.EVT_CHECKBOX, self.on_autoclip_far, self.auto_clip_far)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(auto_clip, 1, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        button_sizer.Add(self.auto_clip_near, 1, wx.LEFT)
        button_sizer.Add(self.auto_clip_far, 1, wx.LEFT)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.opengl_canvas, 1, wx.EXPAND)
        sizer.Add(button_sizer, 0, wx.BOTTOM | wx.LEFT)
        parent.SetSizerAndFit(sizer)
        self.tool_window.manage(placement="right")

    def on_autoclip_near(self, event):
        session = self._session()
        v = session.main_view
        planes = v.clip_planes
        if self.auto_clip_near.IsChecked():
            planes.remove_plane('near')
            return
        p = planes.find_plane('near')
        if p:
            return
        b = v.drawing_bounds()
        if b is None:
            session.logger.info("Can not turn off automatic clipping since there are no models to clip")
            self.auto_clip_near.SetValue(True)
            return
        near, far = v.near_far_distances(v.camera, None)
        camera_pos = v.camera.position.origin()
        vd = v.camera.view_direction()
        plane_point = camera_pos + near * vd
        planes.set_clip_position('near', plane_point, v.camera)

    def on_autoclip_far(self, event):
        session = self._session()
        v = session.main_view
        planes = v.clip_planes
        if self.auto_clip_far.IsChecked():
            planes.remove_plane('far')
            return
        p = planes.find_plane('far')
        if p:
            return
        b = v.drawing_bounds()
        if b is None:
            session.logger.info("Can not turn off automatic clipping since there are no models to clip")
            self.auto_clip_far.SetValue(True)
            return
        near, far = v.near_far_distances(v.camera, None)
        camera_pos = v.camera.position.origin()
        vd = v.camera.view_direction()
        plane_point = camera_pos + far * vd
        planes.set_clip_position('far', plane_point, v.camera)
