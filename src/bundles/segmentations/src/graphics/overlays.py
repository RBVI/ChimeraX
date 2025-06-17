# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from numpy import (
    array,
    zeros,
    float32,
    uint8,
    int32,
    where,
    expand_dims,
    array_equal,
    rot90,
)

from chimerax.map import VolumeSurface
from chimerax.graphics import Drawing
from chimerax.graphics.drawing import rgba_drawing, position_rgba_drawing
from chimerax.graphics.camera import ortho
from chimerax.segmentations.types import Direction, Axis


class SegmentationCursorOverlay(Drawing):
    def __init__(self, name, radius, thickness):
        super().__init__(name)
        self.max_point_size = None
        self.display_style = Drawing.Dot
        self.use_lighting = False
        self.drawing_center = [0, 0]
        self._radius = radius
        self._center = [0, 0, 0]
        self._thickness = thickness

    def draw(self, renderer, draw_pass):
        from chimerax.graphics.opengl import GL

        if not self.max_point_size:
            self.max_point_size = GL.glGetIntegerv(GL.GL_POINT_SIZE_RANGE)[1]
        GL.glPointSize(min(self.max_point_size, self.thickness))
        r = renderer
        ww, wh = r.render_size()
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        )
        GL.glPointSize(1)

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        if self.max_point_size > thickness > 0:
            self._thickness = thickness
        else:
            raise ValueError("Thickness exceeds OpenGL limit")

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        self._center = center

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        if radius <= 0:
            radius = 1
        self._radius = radius

    def update(self):
        vc, v, _, t = self._geometry()
        self.set_geometry(v, _, t)
        self.vertex_colors = vc

    def _geometry(self):
        # Bresenham's Algorithm
        def mirror_points_8(x, y):
            return [
                (x, y),
                (y, x),
                (-x, y),
                (-y, x),
                (x, -y),
                (y, -x),
                (-x, -y),
                (-y, -x),
            ]

        x = 0
        y = self.radius
        d = 1 - y
        v = []
        v.extend(mirror_points_8(x, y))
        while y > x:
            if d < 0:
                d += 2 * x + 3
            else:
                d += 2 * (x - y) + 5
                y -= 1
            x += 1
            v.extend(mirror_points_8(x, y))
        fv = [[self.center[0] + vt[0], self.center[1] + vt[1], 0] for vt in v]
        # We don't use this but we must pass t along so compute it anyway
        t = []
        for i in range(0, len(v)):
            t.append([i, i + 1])
        t[0][1] = 0
        t[-1][1] = 0
        fv = array(fv, dtype=float32)
        t = array(t, dtype=int32)
        vc = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        return vc, fv, None, t


class SegmentationCursorOnOtherAxisOverlay(Drawing):
    """This is the chord formed by the intersection of the segmentation cursor overlay and the
    guideline overlay for some other axis."""

    def __init__(self, name, direction=Direction.VERTICAL):
        super().__init__(name)
        self.display_style = Drawing.Mesh
        self.use_lighting = False
        self.direction = direction
        self.offset = 0
        self.bottom = 0
        self.top = 0
        self.center_on_drawing = 0
        self.length = 0

    def draw(self, renderer, draw_pass):
        r = renderer
        ww, wh = r.render_size()
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        )

    def update(self):
        vc, v, _, t = self._geometry()
        self.set_geometry(v, _, t)
        self.vertex_colors = vc

    def _geometry(self):
        bottom = max(self.bottom, self.center_on_drawing - self.length / 2)
        top = min(self.top, self.center_on_drawing + self.length / 2)
        if self.direction == Direction.VERTICAL:
            v = [
                [self.offset - 1, bottom, 0],
                [self.offset - 1, top, 0],
                [self.offset, top, 0],
                [self.offset, bottom, 0],
                [self.offset + 1, bottom, 0],
                [self.offset + 1, top, 0],
            ]
            t = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
            v = array(v, dtype=float32)
            t = array(t, dtype=int32)
            c = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        else:
            v = [
                [bottom, self.offset, 0],
                [top, self.offset, 0],
                [top, self.offset + 1, 0],
                [bottom, self.offset + 1, 0],
            ]
            t = [[0, 1], [1, 2], [2, 3], [3, 0]]
            v = array(v, dtype=float32)
            t = array(t, dtype=int32)
            c = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        return c, v, None, t


class OrthoplaneLocationOverlay(Drawing):
    """For any axis, this draws a line at the locations of the other two axes' slices."""

    def __init__(self, name, slice, direction=Direction.VERTICAL):
        super().__init__(name)
        self.display_style = Drawing.Mesh
        self.use_lighting = False
        self.direction = direction
        self._slice = slice
        self.bottom = 0
        self.offset = 0
        self.top = 0
        self.tick_thickness = 1

    def draw(self, renderer, draw_pass):
        r = renderer
        ww, wh = r.render_size()
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        )

    def screen_space_offset(self):
        return self.slice * self.tick_thickness + self.offset

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, slice):
        self._slice = slice

    def update(self):
        vc, v, _, t = self._geometry()
        self.set_geometry(v, _, t)
        self.vertex_colors = vc

    def _geometry(self):
        if self.direction == Direction.VERTICAL:
            ofs = (self.slice * self.tick_thickness) + self.offset
            v = [
                [ofs - 1, self.bottom, 0],
                [ofs - 1, self.top, 0],
                [ofs, self.top, 0],
                [ofs, self.bottom, 0],
                [ofs + 1, self.bottom, 0],
                [ofs + 1, self.top, 0],
            ]
            t = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
            v = array(v, dtype=float32)
            t = array(t, dtype=int32)
            c = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        else:
            ofs = (self.slice * self.tick_thickness) + self.offset
            v = [
                [self.bottom, ofs, 0],
                [self.top, ofs, 0],
                [self.top, ofs + 1, 0],
                [self.bottom, ofs + 1, 0],
            ]
            t = [[0, 1], [1, 2], [2, 3], [3, 0]]
            v = array(v, dtype=float32)
            t = array(t, dtype=int32)
            c = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        return c, v, None, t


class SegmentationOverlay(Drawing):
    """This highlights the region of the current slice that is segmented."""

    def __init__(self, name, segmentation, axis) -> None:
        super().__init__(name)
        self.display_style = Drawing.Solid
        self.segmentation = segmentation
        for child in self.segmentation._child_drawings:
            if type(child) == VolumeSurface:
                self.segmentation_surface = child
                self._color = child.color
        self.axis = axis
        self._slice = 0
        self._window_size = None
        self._texture_size = None
        self._texture_pixel_scale = None
        self._aspect = 1
        self._color = None
        self._x_min = 0
        self._x_max = 0
        self._y_min = 0
        self._y_max = 0

    def all_drawings(self, displayed_only=False):
        # Iteratively check parents to see if they are displayed. If any parent model is hidden,
        # return an empty list.
        if not self.segmentation_surface.parent.active:
            return []
        dlist = super().all_drawings(displayed_only=displayed_only)
        parent = self.segmentation_surface
        while parent:
            if not parent.display:
                return []
            parent = getattr(parent, "parent", None)
        return dlist

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, slice):
        if slice != self._slice:
            self._slice = slice
            self.needs_update = True

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def y_min(self):
        return self._y_min

    @property
    def y_max(self):
        return self._y_max

    @x_min.setter
    def x_min(self, x_min):
        if x_min != self._x_min:
            self._x_min = x_min
            self.needs_update = True

    @x_max.setter
    def x_max(self, x_max):
        if x_max != self._x_max:
            self._x_max = x_max
            self.needs_update = True

    @y_min.setter
    def y_min(self, y_min):
        if y_min != self._y_min:
            self._y_min = y_min
            self.needs_update = True

    @y_max.setter
    def y_max(self, y_max):
        if y_max != self._y_max:
            self._y_max = y_max
            self.needs_update = True

    def draw(self, renderer, draw_pass):
        r = renderer
        self._update_graphics(renderer)
        r = renderer
        ww, wh = r.render_size()
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        )

    def _update_graphics(self, renderer):
        """Recompute the overlay if the segmentation slice has changed, or if the
        2D slice has moved."""
        # Get the current segmentation slice and convert it into an RGBA array.
        # Get the color of the volume that owns the segmentation and use that so it looks consistent
        # with the 3D view.
        # The 2dlabels _update_label_image code uses x * y * 4 for the size of the array.
        # The 4 is for RGBA.
        w, h = window_size = renderer.render_size()
        pscale = renderer.pixel_scale()
        aspect = 1
        if pscale != self._texture_pixel_scale or aspect != self._aspect:
            self._texture_pixel_scale = pscale
            self._aspect = aspect
            self.needs_update = True
        if not array_equal(self.segmentation_surface.color, self._color):
            self._color = self.segmentation_surface.color
            self.needs_update = True
        if self.needs_update:
            if self.axis == Axis.AXIAL:
                slice_data = rot90(self.segmentation.data.array[self.slice], 2)
            elif self.axis == Axis.CORONAL:
                slice_data = self.segmentation.data.array[:, self.slice, :]
            else:
                slice_data = self.segmentation.data.array[:, :, self.slice]

            # Convert the segmentation slice into an RGBA array.
            tx, ty = slice_data.shape
            # The volume itself for some reason does not take on the color of its
            # VolumeImage/VolumeSurface
            zero = zeros((tx, ty, 4), dtype=uint8)
            expanded_slice_data = expand_dims(slice_data, -1)
            rgba = where(expanded_slice_data == 0, zero, self._color)
            ih, iw = rgba.shape[:2]
            self.set_transparency(125)
            self._texture_size = (iw, ih)
            x, y = self.x_min, self.y_min
            w, h = self.x_max - x, self.y_max - y
            rgba_drawing(self, rgba, (x, y), (w, h), opaque=False)
            # TODO: The origin needs to be set to the bottom left corner of the
            # texture, not the bottom left corner of the window.
            # TODO: The width and height need to be set to the width and height
            # of the texture, not the width and height of the window.
            position_rgba_drawing(self, (x, y), (w, h))
            self.needs_update = False
