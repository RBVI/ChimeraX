# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from .settings import defaults

from chimerax.core.models import Surface
from chimerax.dist_monitor import SimpleMeasurable, ComplexMeasurable
class PlaneModel(Surface, ComplexMeasurable):
    def __init__(self, session, name, plane, thickness, radius, color):
        super().__init__(name, session)
        self.plane = plane
        self._thickness = thickness
        self.color = color
        self._radius = radius
        self.display_style = self.Solid
        self._update_geometry()

    def angle(self, obj):
        if isinstance(obj, AxisModel):
            return _axis_plane_angle(obj, self)
        elif not isinstance(obj, PlaneModel):
            return NotImplemented
        from chimerax.geometry import angle
        degrees = angle(self.xform_normal, obj.xform_normal)
        return degrees if degrees < 90 else 180 - degrees

    def distance(self, obj, *, signed=False):
        if isinstance(obj, PlaneModel):
            from chimerax.geometry.plane import PlaneNoIntersectionError
            try:
                self.plane.intersection(obj.plane)
            except PlaneNoIntersectionError:
                class CoordMeasurable(SimpleMeasurable):
                    def __init__(self, crd):
                        self.coord = crd

                    @property
                    def scene_coord(self):
                        return self.coord
                return self.distance(CoordMeasurable(obj.position * obj.plane.origin), signed=signed)
            return 0.0
        if not isinstance(obj, SimpleMeasurable):
            return NotImplemented
        scene_crd = obj.scene_coord
        # need to inverse transform to get into same coord sys as self.plane
        signed_dist = self.plane.distance(self.position.inverse() * scene_crd)
        return signed_dist if signed else abs(signed_dist)

    @property
    def normal(self):
        return self.plane.normal

    def _get_radius(self):
        return self._radius

    def _set_radius(self, val):
        if self._radius == val:
            return
        self._radius = val
        self._updated()

    radius = property(_get_radius, _set_radius)

    def _get_thickness(self):
        return self._thickness

    def _set_thickness(self, val):
        if self._thickness == val:
            return
        self._thickness = val
        self._updated()

    thickness = property(_get_thickness, _set_thickness)

    def _updated(self):
        self._update_geometry()
        self.redraw_needed()

    def _update_geometry(self):
        from chimerax.shape.shape import cylinder_geometry
        varray, tarray = cylinder_geometry(self.radius, self.thickness,
            max(2, int(self.thickness / 3 + 0.5)), max(40, int(20.0 * self.radius + 0.5)), True)
        from chimerax.geometry import translation, vector_rotation
        varray = (translation(self.plane.origin)
            * vector_rotation((0,0,1), self.plane.normal)).transform_points(varray)
        from chimerax.surface import calculate_vertex_normals
        narray = calculate_vertex_normals(varray, tarray)
        self.set_geometry(varray, narray, tarray)

    def take_snapshot(self, session, flags):
        return {
            'version': 1,
            'base data': super().take_snapshot(session, flags),
            'plane': self.plane,
            'thickness': self.thickness,
            'radius': self.radius,
            'color': self.color
        }

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = cls(session, None, data['plane'], data['thickness'], data['radius'], data['color'])
        Surface.set_state_from_snapshot(inst, session, data['base data'])
        return inst

    @property
    def xform_normal(self):
        return self.position.transform_vector(self.normal)


class AxisModel(Surface, ComplexMeasurable):
    def __init__(self, session, name, center, direction, extent, radius, color):
        super().__init__(name, session)
        self.color = color
        self._center = center
        self._direction = direction
        self._extent = extent
        self._radius = radius
        self.display_style = self.Solid
        self._update_geometry()

    def angle(self, obj):
        if isinstance(obj, PlaneModel):
            return _axis_plane_angle(self, obj)
        elif not isinstance(obj, AxisModel):
            return NotImplemented
        return quadrant_angle(angle(self.xform_direction, obj.xform_direction))

    def _axis_ends_dist(self, axis):
        return min(
            min([self._point_distance(axis.xform_center + axis.xform_direction * f * axis.extent)
                for f in (-1, 1)]),
            min([axis._point_distance(self.xform_center + self.xform_direction * f * self.extent)
                for f in (-1, 1)])
        )

    @property
    def center(self):
        return self._center

    @property
    def direction(self):
        return self._direction

    def distance(self, obj, *, signed=False):
        if isinstance(obj, AxisModel):
            # shortest distance between lines is perpendicular to both...
            self_dir = self.xform_direction
            obj_dir = obj.xform_direction
            from chimerax.geometry import angle, cross_product, Plane
            if angle(self_dir, obj_dir) in [0.0, 180.0]:
                # parallel
                return self._axis_ends_dist(obj)
            short_dir = cross_product(self_dir, obj_dir)
            # can use analytically shortest dist only if each axis
            # penetrates the plane formed by the other axis and the
            # perpendicular
            for a1, a2 in [(obj, self), (self, obj)]:
                normal = cross_product(a1.xform_direction, short_dir)
                plane = Plane(a1.xform_center, normal=normal)
                d1 = plane.distance(a2.xform_center + a2.xform_direction * a2.extent)
                d2 = plane.distance(a2.xform_center - a2.xform_direction * a2.extent)
                if (d1 < 0.0 and d2 < 0.0) or (d1 > 0.0 and d2 > 0.0):
                    # both ends on same side of plane
                    return self._axis_ends_dist(obj)
            # distance is the difference between the planes that contain the axes and with
            # a normal perpendicular to both
            d1 = Plane(self.xform_center, normal=short_dir).equation()[3]
            d2 = Plane(obj.xform_center, normal=short_dir).equation()[3]
            return abs(d1 - d2)
        if not isinstance(obj, SimpleMeasurable):
            return NotImplemented
        # put into same coord sys as axis
        return self._point_distance(self.position.inverse() * obj.scene_coord)

    @property
    def extent(self):
        return self._extent

    def _point_distance(self, pt):
        # pt should already be in our transformed coord sys
        xf_center = self.xform_center
        xf_direction = self.xform_direction
        min_pt = xf_center - xf_direction * self.extent
        max_pt = xf_center + xf_direction * self.extent
        v = pt - xf_center
        from chimerax.geometry import cross_product, Plane, inner_product, distance
        c1 = cross_product(v, xf_direction)
        import numpy
        if not numpy.any(c1):
            # colinear
            in_plane = pt
        else:
            plane = Plane(xf_center, normal=xf_direction)
            in_plane = plane.nearest(pt)
        pt_ext = inner_product(in_plane - xf_center, xf_direction)
        if pt_ext < -self.extent:
            measure_pt = min_pt
        elif pt_ext > self.extent:
            measure_pt = max_pt
        else:
            measure_pt = in_plane
        return distance(pt, measure_pt)

    def _get_radius(self):
        return self._radius

    def _set_radius(self, val):
        if self._radius == val:
            return
        self._radius = val
        self._updated()

    radius = property(_get_radius, _set_radius)

    def take_snapshot(self, session, flags):
        return {
            'version': 1,
            'base data': super().take_snapshot(session, flags),
            'center': self.center,
            'direction': self.direction,
            'extent': self.extent,
            'radius': self.radius,
            'color': self.color
        }

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = cls(session, None, data['center'], data['direction'], data['extent'], data['radius'],
            data['color'])
        Surface.set_state_from_snapshot(inst, session, data['base data'])
        return inst

    def _update_geometry(self):
        from chimerax.shape.shape import cylinder_geometry
        length = 2 + self.extent
        varray, tarray = cylinder_geometry(self.radius, length,
            max(2, int(length / 3 + 0.5)), max(40, int(20.0 * self.radius + 0.5)), True)
        from chimerax.geometry import translation, vector_rotation
        varray = (translation(self.center)
            * vector_rotation((0,0,1), self.direction)).transform_points(varray)
        from chimerax.surface import calculate_vertex_normals
        narray = calculate_vertex_normals(varray, tarray)
        self.set_geometry(varray, narray, tarray)

    @property
    def xform_center(self):
        return self.position.transform_vector(self.center)

    @property
    def xform_direction(self):
        return self.position.transform_vector(self.direction)

def _axis_plane_angle(axis, plane):
    from chimerax.geometry import angle
    anti_angle = angle(axis.xform_direction, plane.xform_normal)
    return 90.0 - quadrant_angle(anti_angle)

def quadrant_angle(angle):
    while angle < 0.0:
        angle += 360.0
    while angle > 360.0:
        angle -= 360.0
    if angle > 180.0:
        angle = 360.0 - angle
    if angle > 90.0:
        angle = 180.0 - angle
    return angle

def cmd_define_plane(session, atoms, *, thickness=defaults["plane_thickness"], padding=0.0, color=None,
        radius=None, name="plane"):
    """Wrapper to be called by command line.

       Use chimerax.geometry.Plane for other programming applications.
    """
    from chimerax.core.errors import UserError

    from chimerax.atomic import AtomicStructure, concatenate, Structure
    if atoms is None:
        structures_atoms = [m.atoms for m in session.models if isinstance(m, AtomicStructure)]
        if structures_atoms:
            atoms = concatenate(structures_atoms)
        else:
            raise UserError("Atom specifier selects no atoms")
    if len(atoms) < 3:
        raise UserError("Must specify at least 3 atoms to define a plane")

    structures = atoms.unique_structures
    if len(structures) > 1:
        crds = atoms.scene_coords
    else:
        crds = atoms.coords

    from chimerax.geometry import Plane, distance_squared
    plane = Plane(crds)

    if radius is None:
        max_sq_dist = None
        origin = plane.origin
        for crd in crds:
            projected = plane.nearest(crd)
            sq_dist = distance_squared(origin, projected)
            if max_sq_dist is None or sq_dist > max_sq_dist:
                max_sq_dist = sq_dist
        from math import sqrt
        radius = sqrt(max_sq_dist)

    if color is None:
        from chimerax.atomic.colors import element_color, predominant_color
        color = predominant_color(atoms)
        if color is None:
           color = element_color(6)
    else:
        color = color.uint8x4()

    plane_model = PlaneModel(session, name, plane, thickness, radius+padding, color)
    if len(structures) > 1:
        session.models.add([plane_model])
    else:
        structures[0].add([plane_model])
    session.logger.info("Plane '%s' placed at %s with normal %s" % (name, plane.origin, plane.normal))
    return plane_model

def cmd_define_axis(session, axis_info, *, color=None, radius=None, length=None, name=None,
        primary=True, secondary=False, tertiary=False, mass_weighting=False):
    """Wrapper to be called by command line.

       Use chimerax.geometry.vector for other programming applications.
    """
    from chimerax.core.errors import UserError

    from chimerax.atomic.colors import element_color, predominant_color
    from chimerax.core.commands import Axis
    import numpy
    if not (primary or secondary or tertiary):
        raise UserError("One of 'primary', 'secondary', or 'tertiary' must be specified as true.")
    if color is not None:
        color = color.uint8x4()
    axes_info = []
    structure = None
    if isinstance(axis_info, Axis):
        if secondary or tertiary:
            raise UserError("Must specify 3 or more atoms to determine secondary/tertiary axes")
        if mass_weighting:
            raise UserError("Must specify 3 or more atoms for mass weighting")
        base_pt = axis_info.base_point()
        if base_pt is None:
            base_pt = numpy.array([0.0, 0.0, 0.0])
        vec = axis_info.scene_coordinates(normalize=False)
        if color is None:
            if axis_info.atoms is not None:
                color = predominant_color(axis_info.atoms, none_fraction=0.6)
            if color is None:
                color = element_color(6)
        if axis_info.atoms is not None and axis_info.atoms[0].structure == axis_info.atoms[1].structure:
            structure = axis_info.atoms[0]

        center = ((vec + base_pt) + base_pt) / 2
        from chimerax.geometry import distance
        extent = distance(vec + base_pt, base_pt) / 2
        axes_info.append((name, center, v_pt, extent, 1.0 if radius is None else radius, color))
    else:
        if color is None:
            color = predominant_color(axis_info)
            if color is None:
                color = element_color(6)
        us = axis_info.unique_structures
        if len(us) == 1:
            structure = us[0]
        from numpy.linalg import eig, svd, eigh
        if mass_weighting:
            weights = axis_info.elements.masses
            n = len(axis_info)
            mat_weights = weights.reshape((n,1))
            wcoords = mat_weights * axis_info.scene_coords
            wsum = weights.sum()
            centroid = wcoords.sum(0) / wsum
            centered = axis_info.scene_coords - centroid
            _, vals, vecs = svd(mat_weights * centered, full_matrices=False)
        else:
            centroid = axis_info.scene_coords.mean(0)
            centered = axis_info.scene_coords - centroid
            _, vals, vecs = svd(centered, full_matrices=False)
        order = reversed(vals.argsort())
        for index, name, use in zip(order, ('primary', 'secondary', 'tertiary'),
                (primary, secondary, tertiary)):
            if not use:
                continue
            vec = vecs[index]
            if length is None:
                dotted = numpy.dot(centered, vec)
                bounds = (dotted.min(), dotted.max())
                extent = (bounds[1] - bounds[0]) / 2
                center = ((centroid + bounds[0] * vec) + (centroid + bounds[1] * vec)) / 2
            else:
                extent = length / 2
                center = centroid
            if radius is None:
                # average of distances to axis
                ts = numpy.tensordot(vec, centered, (0, 1)) / numpy.dot(vec, vec)
                line_pts = numpy.outer(ts, vec)
                temp = (centered - line_pts)
                r = numpy.sqrt((temp * temp).sum(-1)).mean(0)
            else:
                r = radius
            axes_info.append((name, center, vec, extent, r, color))

    if len(axes_info) > 1:
        from chimerax.core.models import Model
        add_model = Model("axes" if name is None else name, session)
        if structure:
            structure.add([add_model])
        else:
            session.models.add([add_model])
    else:
        add_model = structure
    axes = []
    for axis_name, center, direction, extent, radius, color in axes_info:
        if axis_name is None:
            axis_name = "axis"
        session.logger.info("Axis '%s' centered at %s with direction %s and length %g"
            % (axis_name, center, direction, 2*extent))
        if structure:
            inverse = structure.position.inverse()
            center = inverse * center
            direction = inverse * direction
        axes.append(AxisModel(session, axis_name, center, direction, extent, radius, color))
    if add_model:
        add_model.add(axes)
    else:
        session.models.add(axes)
    return axes


def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, FloatArg, ColorArg, PositiveFloatArg
    from chimerax.core.commands import StringArg, Or, AxisArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('thickness', PositiveFloatArg), ('padding', FloatArg), ('color', ColorArg),
            ('radius', PositiveFloatArg), ('name', StringArg)],
        synopsis = 'Create plane'
    )
    register('define plane', desc, cmd_define_plane, logger=logger)

    desc = CmdDesc(
        required=[('axis_info', Or(AxisArg, AtomsArg))],
        keyword = [('color', ColorArg), ('radius', PositiveFloatArg), ('length', PositiveFloatArg),
            ('name', StringArg), ('primary', BoolArg), ('secondary', BoolArg), ('tertiary', BoolArg),
            ('mass_weighting', BoolArg)],
        synopsis = 'Create plane'
    )
    register('define axis', desc, cmd_define_axis, logger=logger)
