# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

import numpy
from chimerax.graphics import Drawing, Pick
from chimerax.core.state import State
from collections import namedtuple

AtomicShapeInfo = namedtuple(
    "AtomicShapeInfo",
    ("vertices", "normals", "triangles", "color", "atoms", "description"),
    defaults=(None, None))


class AtomicShapeDrawing(Drawing, State):
    """Extend Drawing with knowledge of individual shapes

    The only additional public API is the :py:meth:`add_shape` method.

    If an individual shape corresponds to atom or multiple atoms, then
    when that shape it picked, the atoms are picked too and vice-versa.
    """
    SESSION_VERSION = 1

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._shapes = []
        self._selected_shapes = set()
        self._selection_handler = None

    def take_snapshot(self, session, flags):
        from chimerax.graphics.gsession import DrawingState
        data = {}
        data['version'] = AtomicShapeDrawing.SESSION_VERSION
        data['drawing'] = DrawingState.take_snapshot(self, session, flags)
        data['shapes'] = list((s.triangle_range, s.description, s.atoms) for s in self._shapes)
        data['selected'] = [
            i for i in range(len(self._shapes))
            if self._shapes[i] in self._selected_shapes
        ]
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        from chimerax.graphics.gsession import DrawingState
        d = AtomicShapeDrawing('')
        DrawingState.set_state_from_snapshot(d, session, data['drawing'])
        d._shapes = [_AtomicShape(*args) for args in data['shapes']]
        d._selected_shapes.update([d._shapes[i] for i in data['selected']])
        if any(s.atoms for s in d._shapes):
            # After drawing is added to parent, add back the selection handler
            def post_shape_handler(trigger, trigger_data, drawing=d):
                from chimerax.core.triggerset import DEREGISTER
                drawing._add_handler_if_needed()
                return DEREGISTER
            session.triggers.add_handler("end restore session", post_shape_handler)
        return d

    def delete(self):
        if self._selection_handler:
            self.parent.session.remove_handler(self._selection_handler)
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
        s = _search(self._shapes, tn)
        if s is None:
            return None
        pick = PickedAtomicShape(pick.distance, s, self)
        return pick

    def planes_pick(self, planes, exclude=None):
        if not self.display:
            return []
        if exclude is not None and exclude(self):
            return []

        picks = []
        all_picks = super().planes_pick(planes, exclude)
        from chimerax.graphics import PickedTriangles
        for p in all_picks:
            if not isinstance(p, PickedTriangles) or p.drawing() is not self:
                picks.append(p)
                continue
            tmask = p._triangles_mask
            shapes = [s for s in self._shapes if tmask[s.triangle_range].sum() > 0]
            if shapes:
                picks.append(PickedAtomicShapes(shapes, self))
        return picks

    def selected_items(self, itype):
        if itype in ('atoms', 'bonds'):
            from chimerax.atomic import Atoms
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
        self._selected_shapes = set(s for s in self._shapes if s.atoms and all(s.atoms.selected))
        tmask = self.highlighted_triangles_mask
        if tmask is None:
            tris = self.triangles
            if tris is None:
                return
            tmask = numpy.zeros(len(tris), bool)
        else:
            tmask[:] = False
        for s in self._selected_shapes:
            tmask[s.triangle_range] = True
        self.highlighted_triangles_mask = tmask

    def _add_selected_shape(self, shape):
        self._selected_shapes.add(shape)
        tmask = self.highlighted_triangles_mask
        if tmask is None:
            tmask = numpy.zeros(len(self.triangles), bool)
        tmask[shape.triangle_range] = True
        self.highlighted_triangles_mask = tmask
        if shape.atoms:
            shape.atoms.selected = True

    def _add_selected_shapes(self, shapes):
        self._selected_shapes.update(shapes)
        tmask = self.highlighted_triangles_mask
        if tmask is None:
            tmask = numpy.zeros(len(self.triangles), bool)
        for s in shapes:
            tmask[s.triangle_range] = True
            if s.atoms:
                s.atoms.selected = True
        self.highlighted_triangles_mask = tmask

    def _remove_selected_shape(self, shape):
        self._selected_shapes.remove(shape)
        if shape.atoms:
            shape.atoms.selected = False
        tmask = self.highlighted_triangles_mask
        if tmask is None:
            return
        tmask[shape.triangle_range] = False
        self.highlighted_triangles_mask = tmask

    def _remove_selected_shapes(self, shapes):
        self._selected_shapes.difference_update(shapes)
        tmask = self.highlighted_triangles_mask
        for s in shapes:
            tmask[s.triangle_range] = False
            if s.atoms:
                s.atoms.selected = False
        self.highlighted_triangles_mask = tmask

    def _add_handler_if_needed(self):
        if self._selection_handler is not None:
            return
        from chimerax.atomic import Structure
        if isinstance(self.parent, Structure):
            return
        from chimerax.core.selection import SELECTION_CHANGED
        self.parent.session.triggers.add_handler(SELECTION_CHANGED, lambda *args, s=self: s.update_selection())

    def add_shape(self, vertices, normals, triangles, color, atoms=None, description=None):
        """Add shape to drawing

        Parameters
        ----------
        vertices : :py:class:`numpy.array` of coordinates
        normals : :py:class:`numpy.array` of normals, one per vertex
        triangles : :py:class:`numpy.array` of vertex indices, multiple of 3
        color : either a single 4 element uint8 :py:class:`numpy.array`;
            or an array of those values, one per vertex
        atoms : a sequence of :py:class:`~chimerax.atomic.Atom`s
            or an :py:class:`~chimerax.atomic.Atoms` collection.
        description : a string describing the shape

        The vertices, normals, and triangles can be custom or the results
        from one of the :py:mod:`~chimerax.surface`'s geometry functions.
        If the description is not given, it defaults to a list of the atoms.
        """
        # extend drawing's vertices, normals, vertex_colors, and triangles
        # atoms is a molarray.Atoms collection
        # description is what shows up when hovered over
        asarray = numpy.asarray
        concat = numpy.concatenate
        if not hasattr(color, 'shape'):
            color = asarray(color, dtype=numpy.uint8)
        if color.ndim == 1 or color.shape[0] == 1:
            colors = numpy.empty((vertices.shape[0], 4), dtype=numpy.uint8)
            colors[:] = color
        else:
            assert colors.shape[1] == 4 and colors.shape[0] == vertices.shape[0]
        if self.vertices is None:
            if atoms is not None:
                self._add_handler_if_needed()
            self.set_geometry(asarray(vertices, dtype=numpy.float32),
                              asarray(normals, dtype=numpy.float32),
                              asarray(triangles, dtype=numpy.int32))
            self.vertex_colors = colors
            s = _AtomicShape(range(0, self.triangles.shape[0]), description, atoms)
            self._shapes.append(s)
            return
        offset = self.vertices.shape[0]
        start = self.triangles.shape[0]
        new_vertex_colors = concat((self.vertex_colors, colors))
        self.set_geometry(asarray(concat((self.vertices, vertices)), dtype=numpy.float32),
                          asarray(concat((self.normals, normals)), dtype=numpy.float32),
                          asarray(concat((self.triangles, triangles + offset)), dtype=numpy.int32))
        self.vertex_colors = new_vertex_colors
        s = _AtomicShape(range(start, self.triangles.shape[0]), description, atoms)
        self._shapes.append(s)

    def add_shapes(self, shape_info):
        """Add multiple shapes to drawing

        Parameters
        ----------
        shape_info: sequence of :py:class:`AtomicShapeInfo`

        There must be no initial geometry.
        """
        from numpy import empty, float32, int32, uint8, concatenate as concat
        num_shapes = len(shape_info)
        all_vertices = [None] * num_shapes
        all_normals = [None] * num_shapes
        all_triangles = [None] * num_shapes
        all_colors = [None] * num_shapes
        all_shapes = [None] * num_shapes
        num_vertices = 0
        num_triangles = 0
        has_atoms = False
        for i, info in enumerate(shape_info):
            vertices, normals, triangles, color, atoms, description = info
            all_vertices[i] = vertices
            all_normals[i] = normals
            all_triangles[i] = triangles + num_vertices
            if not hasattr(color, 'shape'):
                color = numpy.asarray(color, dtype=numpy.uint8)
            if color.ndim == 1 or color.shape[0] == 1:
                colors = empty((vertices.shape[0], 4), dtype=uint8)
                colors[:] = color
            else:
                assert colors.shape[1] == 4 and colors.shape[0] == vertices.shape[0]
            all_colors[i] = colors
            has_atoms = has_atoms or (atoms is not None)
            new_num_triangles = num_triangles + len(triangles)
            all_shapes[i] = _AtomicShape(range(num_triangles, new_num_triangles), description, atoms)
            num_vertices += len(vertices)
            num_triangles = new_num_triangles
        if has_atoms:
            self._add_handler_if_needed()
        vertices = empty((num_vertices, 3), dtype=float32)
        normals = empty((num_vertices, 3), dtype=float32)
        triangles = empty((num_triangles, 3), dtype=int32)
        self.set_geometry(
            concat(all_vertices, out=vertices),
            concat(all_normals, out=normals),
            concat(all_triangles, out=triangles))
        self.vertex_colors = concat(all_colors)
        self._shapes = all_shapes

    def extend_shape(self, vertices, normals, triangles, color=None):
        """Extend previous shape

        Parameters
        ----------
        vertices : :py:class:`numpy.array` of coordinates
        normals : :py:class:`numpy.array` of normals, one per vertex
        triangles : :py:class:`numpy.array` of vertex indices, multiple of 3
        color : either None, a single 4 element uint8 :py:class:`numpy.array`;
            or an array of those values, one per vertex.  If None, then the
            color is same as the last color of the existing shape.
        The associated atoms and description are that of the extended shape.

        """
        if self.vertices is None:
            raise ValueError("no shape to extend")
        asarray = numpy.asarray
        concat = numpy.concatenate
        if color is not None and not hasattr(color, 'shape'):
            color = asarray(color, dtype=numpy.uint8)
        if color is None or color.ndim == 1 or color.shape[0] == 1:
            colors = numpy.empty((vertices.shape[0], 4), dtype=numpy.uint8)
            if color is None:
                colors[:] = self.vertex_colors[-1]
            else:
                colors[:] = color
        else:
            assert colors.shape[1] == 4 and colors.shape[0] == vertices.shape[0]
        offset = self.vertices.shape[0]
        new_vertex_colors = concat((self.vertex_colors, colors))
        self.set_geometry(asarray(concat((self.vertices, vertices)), dtype=numpy.float32),
                          asarray(concat((self.normals, normals)), dtype=numpy.float32),
                          asarray(concat((self.triangles, triangles + offset)), dtype=numpy.int32))
        self.vertex_colors = new_vertex_colors
        s = self._shapes[-1]
        s.triangle_range = range(s.triangle_range.start, self.triangles.shape[0])


class _AtomicShape:

    def __init__(self, triangle_range, description, atoms):
        # triangle_range is a range object that corresponds to the indices
        # of the triangles for that shape in the vertex array
        self.triangle_range = triangle_range
        self.description = description
        from chimerax.atomic import Atoms
        if atoms is not None and not isinstance(atoms, Atoms):
            atoms = Atoms(atoms)
        self.atoms = atoms


class PickedAtomicShape(Pick):

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
                d.append("%s@%s" % (r, ','.join(a.name for a in ra[r])))
            return ','.join(d)
        return d

    def drawing(self):
        return self._drawing

    @property
    def id_string(self):
        d = self.drawing()
        return d.id_string if hasattr(d, 'id_string') else '?'

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
        if not self.shape.atoms:
            return  # TODO: allow shapes without atoms to be selected
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


class PickedAtomicShapes(Pick):

    def __init__(self, shapes, drawing):
        Pick.__init__(self)
        self.shapes = shapes
        self._drawing = drawing

    def description(self):
        return '%d shapes' % len(self.shapes)

    def select(self, mode='add'):
        shapes = [s for s in self.shapes if s.atoms]
        if not shapes:
            return  # TODO: allow shapes without atoms to be selected
        drawing = self._drawing
        if mode == 'add':
            drawing._add_selected_shapes(shapes)
        elif mode == 'subtract':
            drawing._remove_selected_shapes(shapes)
        elif mode == 'toggle':
            adding, removing = [], []
            for s in shapes:
                if s in drawing._selected_shapes:
                    removing.append(s)
                else:
                    adding.append(s)
            if removing:
                drawing._remove_selected_shapes(removing)
            if adding:
                drawing._add_selected_shapes(adding)


def _search(shapes, tri):
    # Binary search a list of shapes whose triangle_ranges are
    # ascending, non-overlaping, and consecutive for a triangle.
    # Returns matching shape, or None if not found.
    lo = 0
    hi = len(shapes)
    while lo < hi:
        mid = (lo + hi) // 2
        tr = shapes[mid].triangle_range
        if tri < tr.start:
            hi = mid
        elif tri >= tr.stop:
            lo = mid + 1
        else:
            return shapes[mid]
    return None
