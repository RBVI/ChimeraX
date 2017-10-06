import numpy
from chimerax.core import generic3d
from chimerax.core.graphics import Pick


class _Shape:

    def __init__(self, triangle_range, description, atom_spec):
        # triangle_range is a range object that corresponds to the indices
        # of the triangles for that shape in the vertex array
        self.triangle_range = triangle_range
        self.description = description
        self.atom_spec = atom_spec


class _ShapePick(Pick):

    def __init__(self, distance, shape, drawing):
        super().__init__(distance)
        self.shape = shape
        self._drawing = drawing

    def description(self):
        return self.shape.description

    def drawing(self):
        return self._drawing

    def id_string(self):
        d = self.drawing()
        return d.id_string() if hasattr(d, 'id_string') else '?'

    def is_transparent(self):
        d = self.drawing()
        vc = d.vertex_colors
        if vc is None:
            return d.color[3] < 255
        t = self.triangle_num
        for v in d.triangle_range[t]:
            if vc[v, 3] < 255:
                return True
        return False

    def select(self, mode='add'):
        drawing = self._drawing
        if mode == 'add':
            drawing._add_selected_shape(self.shape)
        elif mode == 'subtract':
            drawing._remove_selected_shapes(self.shape)
        elif mode == 'toggle':
            if self.shape in drawing._selected_shapes:
                drawing._remove_selected_shape(self.shape)
            else:
                drawing._add_selected_shape(self.shape)


class ShapeModel(generic3d.Generic3DModel):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._shapes = []
        self._selected_shapes = set()

    def first_intercept(self, mxyz1, mxyz2, exclude=None):
        pick = super().first_intercept(mxyz1, mxyz2, exclude)
        if pick is None:
            return pick
        try:
            tn = pick.triangle_number
        except AttributeError:
            return pick
        for s in self._shapes:
            if tn in s.triangle_range:
                pick = _ShapePick(pick.distance, s, self)
                break
        return pick

    def selected_items(self, itype):
        if itype == 'atoms':
            return [s for s in self._selected_shapes if s.atom_spec is not None]
        return list(self._selected_shapes)

    def _add_selected_shape(self, shape):
        self._selected_shapes.add(shape)
        tmask = self.selected_triangles_mask
        if tmask is None:
            tmask = numpy.zeros(len(self.triangles), bool)
        tmask[shape.triangle_range] = True
        self.selected_triangles_mask = tmask

    def _remove_selected_shape(self, shape):
        self._selected_shapes.remove(shape)
        tmask = self.selected_triangles_mask
        tmask[shape.triangle_range] = False
        self.selected_triangles_mask = tmask

    def add_shape(self, vertices, normals, triangles, color, atom_spec=None, balloon_text=None):
        # extend drawing's vertices, et. al.
        asarray = numpy.asarray
        concat = numpy.concatenate
        color = (numpy.array([color]) * 255).astype(numpy.uint8)
        colors = concat((color,) * vertices.shape[0])
        if self.vertices is None:
            self.vertices = asarray(vertices, dtype=numpy.float32)
            self.normals = asarray(normals, dtype=numpy.float32)
            self.triangles = asarray(triangles, dtype=numpy.int32)
            self.vertex_colors = asarray(colors, dtype=numpy.uint8)
            s = _Shape(range(0, self.triangles.shape[0]), balloon_text, atom_spec)
            self._shapes.append(s)
            return
        offset = self.vertices.shape[0]
        start = self.triangles.shape[0]
        self.vertices = asarray(concat((self.vertices, vertices)), dtype=numpy.float32)
        self.normals = asarray(concat((self.normals, normals)), dtype=numpy.float32)
        self.triangles = asarray(concat((self.triangles, triangles + offset)), dtype=numpy.int32)
        self.vertex_colors = asarray(concat((self.vertex_colors, colors)), dtype=numpy.uint8)
        s = _Shape(range(start, self.triangles.shape[0]), balloon_text, atom_spec)
        self._shapes.append(s)
