# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""Semantic primitive batches and their transient triangle export meshes.

These batches are retained render data, not tessellated geometry generators.
ChimeraX's existing ``sphere_triangulation`` and ``cylinder_geometry`` helpers
are used only when a batch is converted to triangles for export.
"""


class ExportMesh:
    """One local-coordinate mesh batch produced for a scene exporter."""

    __slots__ = ('vertices', 'normals', 'triangles', 'vertex_colors',
                 'texture_coordinates')

    def __init__(self, vertices, normals, triangles, vertex_colors=None,
                 texture_coordinates=None):
        self.vertices = vertices
        self.normals = normals
        self.triangles = triangles
        self.vertex_colors = vertex_colors
        self.texture_coordinates = texture_coordinates


class ExportGeometryContext:
    """Per-export quality settings and temporary prototype cache."""

    def __init__(self, session=None, batch_vertex_limit=250000):
        self.session = session
        self.batch_vertex_limit = max(1, int(batch_vertex_limit))
        self.prototype_cache = {}
        self._scene_primitive_counts = None

    def sphere_triangles(self, count):
        lod = self._level_of_detail()
        count = max(count, self._primitive_count('spheres'))
        return lod.atom_sphere_triangles(count) if lod is not None else 200

    def cylinder_sides(self, count):
        lod = self._level_of_detail()
        count = max(count, self._primitive_count('cylinders'))
        return max(3, lod.bond_cylinder_triangles(count) // 4) if lod is not None else 15

    def pseudobond_sides(self):
        lod = self._level_of_detail()
        return max(3, lod.pseudobond_sides) if lod is not None else 10

    def _level_of_detail(self):
        if self.session is None:
            return None
        try:
            from chimerax.atomic import level_of_detail
            return level_of_detail(self.session)
        except (ImportError, AttributeError):
            return None

    def _primitive_count(self, kind):
        if self.session is None:
            return 0
        counts = self._scene_primitive_counts
        if counts is None:
            counts = {'spheres': 0, 'cylinders': 0}
            seen = set()
            for model in self.session.models.list():
                if not getattr(model, 'visible', False):
                    continue
                for drawing in model.all_drawings(displayed_only=True):
                    if drawing in seen:
                        continue
                    seen.add(drawing)
                    batch = getattr(drawing, 'primitive_batch', None)
                    if batch is not None:
                        if batch.kind == 'cylinders' and batch.sides is not None:
                            continue
                        counts[batch.kind] += batch.quality_count
            self._scene_primitive_counts = counts
        return counts[kind]

    def sphere_prototype(self, triangle_count):
        key = ('sphere', int(triangle_count))
        geometry = self.prototype_cache.get(key)
        if geometry is None:
            from chimerax.geometry.sphere import sphere_triangulation
            vertices, triangles = sphere_triangulation(triangle_count)
            geometry = (vertices, vertices.copy(), triangles)
            self.prototype_cache[key] = geometry
        return geometry

    def cylinder_prototype(self, sides, caps):
        key = ('cylinder', int(sides), tuple(bool(c) for c in caps))
        geometry = self.prototype_cache.get(key)
        if geometry is None:
            # The surface generator supports either both caps or neither.  For a
            # one-ended cylinder, remove the unwanted cap triangles and vertices.
            from chimerax.surface import cylinder_geometry
            both_caps = bool(caps[0] or caps[1])
            vertices, normals, triangles = cylinder_geometry(
                nc=sides, caps=both_caps, height=1)
            if both_caps and caps[0] != caps[1]:
                side_triangle_count = 2 * sides
                cap_offset = side_triangle_count + (0 if caps[0] else sides)
                keep = list(range(side_triangle_count)) + list(range(cap_offset, cap_offset + sides))
                triangles = triangles[keep]
            geometry = (vertices, normals, triangles)
            self.prototype_cache[key] = geometry
        return geometry


class PrimitiveBatch:
    """Base class for retained semantic render primitives."""

    kind = None

    def __init__(self, colors, picking_ids=None):
        from numpy import asarray, arange, int32, uint8
        colors = asarray(colors, dtype=uint8)
        if colors.ndim != 2 or colors.shape[1] != 4:
            raise ValueError('Primitive colors must be an N by 4 array')
        self.colors = colors
        count = len(colors)
        if picking_ids is None:
            picking_ids = arange(count, dtype=int32)
        else:
            picking_ids = asarray(picking_ids, dtype=int32)
        if picking_ids.shape != (count,):
            raise ValueError('Primitive picking ids must have one entry per primitive')
        if (picking_ids < 0).any():
            raise ValueError('Primitive picking ids cannot be negative')
        self.picking_ids = picking_ids

    @property
    def primitive_count(self):
        return len(self.colors)

    @property
    def quality_count(self):
        """Source-object count used by scene level-of-detail calculations."""
        return self.primitive_count

    def primitive_mask(self, source_mask):
        if source_mask is None:
            return None
        return source_mask[self.picking_ids]


class SpherePrimitiveBatch(PrimitiveBatch):
    """Centers, radii, colors, and picking IDs for analytic spheres."""

    kind = 'spheres'

    def __init__(self, centers, radii, colors, picking_ids=None):
        from numpy import arange, asarray, float32, int32
        centers = asarray(centers, dtype=float32)
        radii = asarray(radii, dtype=float32)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError('Sphere centers must be an N by 3 array')
        if radii.shape != (len(centers),):
            raise ValueError('Sphere radii must have one entry per center')
        if len(colors) != len(centers):
            raise ValueError('Sphere colors must have one entry per center')
        if (radii < 0).any():
            raise ValueError('Sphere radii cannot be negative')
        count = len(centers)
        if picking_ids is None:
            picking_ids = arange(count, dtype=int32)
        else:
            picking_ids = asarray(picking_ids)
            if picking_ids.shape != (count,):
                raise ValueError('Sphere picking ids must have one entry per sphere')
        valid = radii > 0
        centers, radii = centers[valid], radii[valid]
        colors = asarray(colors)[valid]
        picking_ids = picking_ids[valid]
        from numpy import empty
        self._instance_parameters = empty((len(centers), 4), float32)
        self._instance_parameters[:, :3] = centers
        self._instance_parameters[:, 3] = radii
        self.centers = self._instance_parameters[:, :3]
        self.radii = self._instance_parameters[:, 3]
        super().__init__(colors, picking_ids)

    @property
    def instance_parameters(self):
        return self._instance_parameters

    def bounds(self):
        if self.primitive_count == 0:
            return None
        radii = self.radii[:, None]
        from chimerax.geometry import Bounds
        return Bounds((self.centers - radii).min(axis=0),
                      (self.centers + radii).max(axis=0))

    def export_geometry(self, context):
        triangle_count = context.sphere_triangles(self.primitive_count)
        unit_v, unit_n, unit_t = context.sphere_prototype(triangle_count)
        vertices_per = len(unit_v)
        per_batch = max(1, context.batch_vertex_limit // vertices_per)
        from numpy import arange, broadcast_to, int32, repeat
        for first in range(0, self.primitive_count, per_batch):
            last = min(first + per_batch, self.primitive_count)
            count = last - first
            vertices = (self.centers[first:last, None, :] +
                        self.radii[first:last, None, None] * unit_v[None, :, :])
            vertices = vertices.reshape((count * vertices_per, 3))
            normals = broadcast_to(unit_n, (count,) + unit_n.shape).reshape(vertices.shape).copy()
            offsets = arange(count, dtype=int32)[:, None, None] * vertices_per
            triangles = (unit_t[None, :, :] + offsets).reshape((-1, 3))
            colors = repeat(self.colors[first:last], vertices_per, axis=0)
            yield ExportMesh(vertices, normals, triangles, colors)


class CylinderPrimitiveBatch(PrimitiveBatch):
    """Endpoints, radii, colors, caps, and IDs for analytic cylinders."""

    kind = 'cylinders'

    def __init__(self, starts, ends, radii, colors, caps=False, picking_ids=None,
                 source_centers=None, sides=None):
        from numpy import asarray, bool_, float32
        starts = asarray(starts, dtype=float32)
        ends = asarray(ends, dtype=float32)
        radii = asarray(radii, dtype=float32)
        count = len(starts)
        if starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape:
            raise ValueError('Cylinder starts and ends must be N by 3 arrays')
        if radii.shape != (count,):
            raise ValueError('Cylinder radii must have one entry per cylinder')
        if len(colors) != count:
            raise ValueError('Cylinder colors must have one entry per cylinder')
        if (radii < 0).any():
            raise ValueError('Cylinder radii cannot be negative')
        if isinstance(caps, (bool, bool_)):
            from numpy import empty
            cap_array = empty((count, 2), dtype=bool_)
            cap_array[:] = caps
        else:
            cap_array = asarray(caps, dtype=bool_)
        if cap_array.shape != (count, 2):
            raise ValueError('Cylinder caps must be a bool or an N by 2 array')
        valid = (((ends - starts) ** 2).sum(axis=1) > 1e-14) & (radii > 0)
        from numpy import empty
        valid_starts, valid_ends, valid_radii = starts[valid], ends[valid], radii[valid]
        self._instance_parameters = empty((len(valid_starts), 4), float32)
        self._instance_parameters[:, :3] = valid_starts
        self._instance_parameters[:, 3] = valid_radii
        self.starts = self._instance_parameters[:, :3]
        self.radii = self._instance_parameters[:, 3]
        self.caps = cap_array[valid]
        self._endpoint_parameters = empty((len(valid_ends), 4), float32)
        self._endpoint_parameters[:, :3] = valid_ends
        self._endpoint_parameters[:, 3] = (self.caps[:, 0].astype(float32) +
                                           2 * self.caps[:, 1].astype(float32))
        self.ends = self._endpoint_parameters[:, :3]
        if picking_ids is None:
            from numpy import arange, int32
            picking_ids = arange(count, dtype=int32)
        else:
            picking_ids = asarray(picking_ids)
            if picking_ids.shape != (count,):
                raise ValueError('Cylinder picking ids must have one entry per cylinder')
        filtered_ids = picking_ids[valid]
        self.source_centers = None if source_centers is None else asarray(source_centers, dtype=float32)
        if self.source_centers is not None:
            if self.source_centers.ndim != 2 or self.source_centers.shape[1] != 3:
                raise ValueError('Cylinder source centers must be an N by 3 array')
            if filtered_ids is not None and len(filtered_ids) and filtered_ids.max() >= len(self.source_centers):
                raise ValueError('Cylinder picking id exceeds source center count')
        if sides is not None:
            sides = int(sides)
            if sides < 3:
                raise ValueError('Cylinder side count must be at least 3')
        self.sides = sides
        super().__init__(asarray(colors)[valid], filtered_ids)

    @property
    def instance_parameters(self):
        return self._instance_parameters

    @property
    def endpoints(self):
        return self._endpoint_parameters

    @property
    def quality_count(self):
        from numpy import unique
        return len(unique(self.picking_ids))

    def bounds(self):
        if self.primitive_count == 0:
            return None
        radii = self.radii[:, None]
        from numpy import minimum, maximum
        xyz_min = minimum(self.starts, self.ends) - radii
        xyz_max = maximum(self.starts, self.ends) + radii
        from chimerax.geometry import Bounds
        return Bounds(xyz_min.min(axis=0), xyz_max.max(axis=0))

    def export_geometry(self, context):
        sides = self.sides
        if sides is None:
            sides = context.cylinder_sides(self.quality_count)
        from numpy import nonzero
        for caps in ((False, False), (True, False), (False, True), (True, True)):
            indices = nonzero((self.caps[:, 0] == caps[0]) &
                              (self.caps[:, 1] == caps[1]))[0]
            if len(indices) == 0:
                continue
            unit_v, unit_n, unit_t = context.cylinder_prototype(sides, caps)
            vertices_per = len(unit_v)
            per_batch = max(1, context.batch_vertex_limit // vertices_per)
            for first in range(0, len(indices), per_batch):
                selection = indices[first:first + per_batch]
                mesh = self._cylinder_mesh(selection, unit_v, unit_n, unit_t)
                if mesh is not None:
                    yield mesh

    def _cylinder_mesh(self, indices, unit_v, unit_n, unit_t):
        from numpy import (abs, arange, broadcast_to, cross, einsum, float32,
                           int32, repeat, stack)
        starts, ends = self.starts[indices], self.ends[indices]
        axes = ends - starts
        lengths = (axes * axes).sum(axis=1) ** 0.5
        valid = lengths > 1e-7
        if not valid.all():
            indices = indices[valid]
            starts, ends, axes, lengths = starts[valid], ends[valid], axes[valid], lengths[valid]
        count = len(indices)
        if count == 0:
            return None
        w = axes / lengths[:, None]
        refs = broadcast_to((0, 0, 1), (count, 3)).astype(float32).copy()
        refs[abs(w[:, 2]) > .9] = (0, 1, 0)
        u = cross(refs, w)
        u /= ((u * u).sum(axis=1) ** .5)[:, None]
        v = cross(w, u)
        radii = self.radii[indices]
        basis = (u * radii[:, None], v * radii[:, None], w * lengths[:, None])
        transform = stack(basis, axis=2)
        midpoints = .5 * (starts + ends)
        vertices = einsum('nij,nvj->nvi', transform, unit_v) + midpoints[:, None, :]
        normal_basis = stack((u, v, w), axis=2)
        normals = einsum('nij,nvj->nvi', normal_basis, unit_n)
        vertices_per = len(unit_v)
        offsets = arange(count, dtype=int32)[:, None, None] * vertices_per
        triangles = (unit_t[None, :, :] + offsets).reshape((-1, 3))
        colors = repeat(self.colors[indices], vertices_per, axis=0)
        return ExportMesh(vertices.reshape((-1, 3)), normals.reshape((-1, 3)),
                          triangles, colors)
