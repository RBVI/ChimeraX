# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from numpy import array, float32, uint8, int32


from chimerax.graphics import Drawing
from chimerax.graphics.camera import ortho
from chimerax.graphics.opengl import GL
from ..types import Direction

class OrthoplaneLocationOverlay(Drawing):
    def __init__(self, name, slice, direction = Direction.VERTICAL):
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
        r.set_projection_matrix((
            (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
        ))

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

    # TODO: Depend on the slice location
    def _geometry(self):
        if self.direction == Direction.VERTICAL:
            ofs = (self.slice * self.tick_thickness) + self.offset
            v = [[ofs-1, self.bottom, 0], [ofs-1, self.top, 0], [ofs, self.top, 0], [ofs, self.bottom, 0], [ofs+1, self.bottom, 0], [ofs+1, self.top, 0]]
            t = [[0, 1], [1,2], [2,3], [3,4], [4,5], [5,0]]
            v = array(v, dtype=float32)
            t = array(t, dtype=int32)
            c = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        else:
            ofs = (self.slice * self.tick_thickness) + self.offset
            v = [[self.bottom, ofs, 0], [self.top, ofs, 0], [self.top ,ofs + 1,0], [self.bottom,ofs+1,0]]
            t = [[0, 1], [1, 2], [2, 3], [3, 0]]
            v = array(v, dtype=float32)
            t = array(t, dtype=int32)
            c = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        return c, v, None, t

class SegCursorOnOtherAxisOverlay(Drawing):
    """This is the chord formed by the intersection of the segmentation cursor overlay and the
    guideline overlay for some other axis."""
    def __init__(self, name):
        super().__init__(name)
        self.display_style = Drawing.Mesh

    def draw(self, renderer, draw_pass):
        r = renderer
        ww, wh = r.render_size()
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
             (0, 0, 0, 1))
        )

    def update(self):
        vc, v, _, t = self._geometry()
        self.set_geometry(v, _, t)
        self.vertex_colors = vc

    def _geometry(self):
        ...

class SegmentationOverlay(Drawing):
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
        if not self.max_point_size:
            self.max_point_size = GL.glGetIntegerv(GL.GL_POINT_SIZE_RANGE)[1]
        GL.glPointSize(min(self.max_point_size, self.thickness))
        r = renderer
        ww, wh = r.render_size()
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
             (0, 0, 0, 1))
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
            return [(x, y), (y, x), (-x, y), (-y, x), (x, -y), (y, -x), (-x, -y), (-y, -x)]
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
