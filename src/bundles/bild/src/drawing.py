# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2017 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

import numpy
from chimerax.core.graphics import Drawing, Pick


class _Shape:

    def __init__(self, triangle_range, description, atoms):
        # triangle_range is a range object that corresponds to the indices
        # of the triangles for that shape in the vertex array
        self.triangle_range = triangle_range
        self.description = description
        from chimerax.core.atomic import Atoms
        if atoms is not None and not isinstance(atoms, Atoms):
            atoms = Atoms(atoms)
        self.atoms = atoms


class PickedShape(Pick):

    def __init__(self, distance, shape, drawing):
        super().__init__(distance)
        self.shape = shape
        self._drawing = drawing

    def description(self):
        d = self.shape.description
        if d is None and self.shape.atoms:
            from collections import OrderedDict
            ra = OrderedDict()
            for a in self.shape.atoms:
                ra.setdefault(a.residue, []).append(a)
            d = []
            for r in ra:
                d.append("%s@%s" % (r.atomspec(), ','.join(a.name for a in ra[r])))
            return ','.join(d)
        return d

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
            drawing._remove_selected_shape(self.shape)
        elif mode == 'toggle':
            if self.shape in drawing._selected_shapes:
                drawing._remove_selected_shape(self.shape)
            else:
                drawing._add_selected_shape(self.shape)


class PickedShapes:

    def __init__(self, shapes, drawing):
        Pick.__init__(self)
        self.shapes = shapes
        self._drawing = drawing

    def description(self):
        return '%d shapes' % len(self.shapes)

    def select(self, mode = 'add'):
        drawing = self._drawing
        if mode == 'add':
            drawing._add_selected_shapes(self.shapes)
        elif mode == 'subtract':
            drawing._remove_selected_shapes(self.shapes)
        elif mode == 'toggle':
            adding, removing = [], []
            for s in self.shapes:
                if s in drawing._selected_shapes:
                    removing.append(s)
                else:
                    adding.append(s)
            if removing:
                drawing._remove_selected_shapes(removing)
            if adding:
                drawing._add_selected_shapes(adding)


class ShapeDrawing(Drawing):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._shapes = []
        self._selected_shapes = set()
        self._selection_handler = None

    def delete(self):
        if self._selection_handler:
            self.session.remove_handler(self._selection_handler)
            self._selection_handler = None
        self._selected_shapes.clear()
        self._shapes.clear()
        super().delete()

    def _first_intercept_excluding_children(self, mxyz1, mxyz2):
        pick = super()._first_intercept_excluding_children(mxyz1, mxyz2)
        if pick is None:
            return None
        try:
            tn = pick.triangle_number
        except AttributeError:
            return pick
        for s in self._shapes:
            if tn in s.triangle_range:
                pick = PickedShape(pick.distance, s, self)
                break
        return pick

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and exclude(self):
            return []

        picks = []
        all_picks = super().planes_pick(planes, exclude)
        from chimerax.core.graphics import PickedTriangles
        for p in all_picks:
            if not isinstance(p, PickedTriangles) or p.drawing() is not self:
                picks.append(p)
                continue
            tmask = p._triangles_mask
            shapes = [s for s in self._shapes if tmask[s.triangle_range].sum() > 0]
            if shapes:
                picks.append(PickedShapes(shapes, self))
        return picks

    def selected_items(self, itype):
        if itype in ('atoms', 'bonds'):
            from chimerax.core.atomic import Atoms
            atoms = Atoms(None)
            for s in self._selected_shapes:
                a = s.atoms
                if a is not None:
                    atoms |= s
            if itype == 'bonds':
                return atoms.intra_bonds
            return atoms
        elif itype == 'shapes':
            return list(self._selected_shapes)
        return []

    def clear_selection(self, include_children=True):
        self._selected_shapes.clear()
        super().clear_selection(include_children=include_children)

    def update_selection(self):
        # called by Structure._update_if_needed when atom selection has changed
        # in a child model/drawing
        self._selected_shapes = set(s for s in self._shapes if s.atoms and any(s.atoms.selected))
        tmask = self.selected_triangles_mask
        if tmask is None:
            tmask = numpy.zeros(len(self.triangles), bool)
        else:
            tmask[:] = False
        for s in self._selected_shapes:
            tmask[s.triangle_range] = True
        self.selected_triangles_mask = tmask

    def _add_selected_shape(self, shape):
        self._selected_shapes.add(shape)
        tmask = self.selected_triangles_mask
        if tmask is None:
            tmask = numpy.zeros(len(self.triangles), bool)
        tmask[shape.triangle_range] = True
        self.selected_triangles_mask = tmask

    def _add_selected_shapes(self, shapes):
        self._selected_shapes.update(shapes)
        tmask = self.selected_triangles_mask
        if tmask is None:
            tmask = numpy.zeros(len(self.triangles), bool)
        for s in shapes:
            tmask[s.triangle_range] = True
        self.selected_triangles_mask = tmask

    def _remove_selected_shape(self, shape):
        self._selected_shapes.remove(shape)
        tmask = self.selected_triangles_mask
        if tmask is None:
            return
        tmask[shape.triangle_range] = False
        self.selected_triangles_mask = tmask

    def _remove_selected_shapes(self, shapes):
        self._selected_shapes.difference_update(shapes)
        tmask = self.selected_triangles_mask
        for s in shapes:
            tmask[s.triangle_range] = False
        self.selected_triangles_mask = tmask

    def _add_handler_if_needed(self):
        if self._selection_handler is not None:
            return
        from chimerax.core.atomic import Structure
        if hasattr(self, 'parent') and isinstance(self.parent, Structure):
            return
        from chimerax.core.selection import SELECTION_CHANGED
        self.session.triggers.add_handler(SELECTION_CHANGED, self.update_selection)

    def add_shape(self, vertices, normals, triangles, color, atoms=None, balloon_text=None):
        # extend drawing's vertices, normals, vertex_colors, and triangles
        # atoms is a molarray.Atoms collection
        # balloon_text is what shows up when hovered over
        if atoms is not None:
            self._add_handler_if_needed()
        asarray = numpy.asarray
        concat = numpy.concatenate
        if color.ndim == 1 or color.shape[0] == 1:
            colors = numpy.empty((vertices.shape[0], 4), dtype=numpy.uint8)
            colors[:] = color
        else:
            colors = color.asarray(color, dtype=numpy.uint8)
            assert colors.shape[1] == 4 and colors.shape[0] == vertices.shape[0]
        if self.vertices is None:
            self.vertices = asarray(vertices, dtype=numpy.float32)
            self.normals = asarray(normals, dtype=numpy.float32)
            self.triangles = asarray(triangles, dtype=numpy.int32)
            self.vertex_colors = colors
            s = _Shape(range(0, self.triangles.shape[0]), balloon_text, atoms)
            self._shapes.append(s)
            return
        offset = self.vertices.shape[0]
        start = self.triangles.shape[0]
        self.vertices = asarray(concat((self.vertices, vertices)), dtype=numpy.float32)
        self.normals = asarray(concat((self.normals, normals)), dtype=numpy.float32)
        self.triangles = asarray(concat((self.triangles, triangles + offset)), dtype=numpy.int32)
        self.vertex_colors = concat((self.vertex_colors, colors))
        s = _Shape(range(start, self.triangles.shape[0]), balloon_text, atoms)
        self._shapes.append(s)
