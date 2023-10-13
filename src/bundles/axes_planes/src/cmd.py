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

from .settings import defaults

from chimerax.core.commands import ModelsArg, AnnotationError
class AxisModelsArg(ModelsArg):
    """Parse command specifier for AxisModels"""
    name = "axis models"

    @classmethod
    def parse(cls, text, session):
        models, text, rest = super().parse(text, session)
        return [m for m in models if isinstance(m, AxisModel)], text, rest

class AxisModelArg(AxisModelsArg):
    """Parse command specifier for an AxisModel"""
    name = "an axis model"

    @classmethod
    def parse(cls, text, session):
        axes, text, rest = super().parse(text, session)
        if len(axes) != 1:
            raise AnnotationError("Must specify 1 axis, got %d" % len(axes), len(text))
        return axes[0], text, rest

class PlaneModelsArg(ModelsArg):
    """Parse command specifier for PlaneModels"""
    name = "plane models"

    @classmethod
    def parse(cls, text, session):
        models, text, rest = super().parse(text, session)
        return [m for m in models if isinstance(m, PlaneModel)], text, rest

class PlaneModelArg(PlaneModelsArg):
    """Parse command specifier for an PlaneModel"""
    name = "an plane model"

    @classmethod
    def parse(cls, text, session):
        axes, text, rest = super().parse(text, session)
        if len(axes) != 1:
            raise AnnotationError("Must specify 1 plane, got %d" % len(axes), len(text))
        return axes[0], text, rest

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

    @property
    def alignment_points(self):
        xform_center = self.scene_position.transform_vector(self.plane.origin)
        xform_normal = self.xform_normal
        return xform_center + xform_normal, xform_center - xform_normal

    def angle(self, obj):
        if isinstance(obj, AxisModel):
            return _axis_plane_angle(obj, self)
        elif not isinstance(obj, PlaneModel):
            return NotImplemented
        from chimerax.geometry import angle
        degrees = angle(self.xform_normal, obj.xform_normal)
        return degrees if degrees < 90 else 180 - degrees

    @property
    def center(self):
        return self.plane.origin
    origin = center

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
                return self.distance(CoordMeasurable(obj.scene_position * obj.plane.origin), signed=signed)
            return 0.0
        if isinstance(obj, AxisModel):
            return _axis_plane_distance(obj, self)
        if not isinstance(obj, SimpleMeasurable):
            return NotImplemented
        scene_crd = obj.scene_coord
        # need to inverse transform to get into same coord sys as self.plane
        signed_dist = self.plane.distance(self.scene_position.inverse() * scene_crd)
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
    def xform_center(self):
        return self.scene_position.transform_vector(self.center)
    xform_origin = xform_center

    @property
    def xform_normal(self):
        return self.scene_position.transform_vector(self.normal)


class AxisModel(Surface, ComplexMeasurable):
    def __init__(self, session, name, center, direction, extent, radius, color, *, needs_normalization=True):
        super().__init__(name, session)
        self.color = color
        self._center = center
        if needs_normalization:
            from chimerax.geometry import normalize_vector
            direction = normalize_vector(direction)
        self._direction = direction
        self._extent = extent
        self._radius = radius
        self.display_style = self.Solid
        self._update_geometry()

    @property
    def alignment_points(self):
        xform_center = self.xform_center
        xform_direction = self.xform_direction * self.extent
        return xform_center + xform_direction, xform_center - xform_direction

    def angle(self, obj):
        if isinstance(obj, PlaneModel):
            return _axis_plane_angle(self, obj)
        elif not isinstance(obj, AxisModel):
            return NotImplemented
        from chimerax.geometry import angle
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
        if isinstance(obj, PlaneModel):
            return _axis_plane_distance(self, obj)
        if not isinstance(obj, SimpleMeasurable):
            return NotImplemented
        # put into same coord sys as axis
        return self._point_distance(self.scene_position.inverse() * obj.scene_coord)

    @property
    def extent(self):
        return self._extent

    @property
    def length(self):
        return 2 * self._extent

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
            from_pt, to_pt = min_pt, pt
        elif pt_ext > self.extent:
            from_pt, to_pt = max_pt, pt
        else:
            from_pt, to_pt = xf_center, in_plane
        return distance(from_pt, to_pt)

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
        length = 2 * self.extent
        varray, tarray = cylinder_geometry(self.radius, length,
            max(2, int(length / 3 + 0.5)), max(40, int(20.0 * self.radius + 0.5)), True)
        from chimerax.geometry import translation, vector_rotation
        varray = (translation(self.center)
            * vector_rotation((0,0,1), self.direction)).transform_points(varray)
        from chimerax.surface import calculate_vertex_normals
        narray = calculate_vertex_normals(varray, tarray)
        self.set_geometry(varray, narray, tarray)

    def _updated(self):
        self._update_geometry()
        self.redraw_needed()

    @property
    def xform_center(self):
        return self.scene_position.transform_vector(self.center)

    @property
    def xform_direction(self):
        return self.scene_position.transform_vector(self.direction)

def _axis_plane_angle(axis, plane):
    from chimerax.geometry import angle
    anti_angle = angle(axis.xform_direction, plane.xform_normal)
    return 90.0 - quadrant_angle(anti_angle)

def _axis_plane_distance(axis, plane):
    # get axis end points in the same reference frame as the plane...
    xform = axis.scene_position.inverse()
    end1 = xform * (axis.center + axis.extent * axis.direction)
    end2 = xform * (axis.center - axis.extent * axis.direction)
    class FakeMeasurable(SimpleMeasurable):
        def __init__(self, crd):
            self._crd = crd

        @property
        def scene_coord(self):
            return self._crd
    d1 = plane.distance(FakeMeasurable(end1), signed=True)
    d2 = plane.distance(FakeMeasurable(end2), signed=True)
    if (d1 < 0 and d2 > 0) or (d2 < 0 and d1 > 0):
        return 0.0
    return min(abs(d1), abs(d2))

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
        radius=None, name="plane", show_tool=True):
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
        radius = sqrt(max_sq_dist) + padding

    if color is None:
        from chimerax.atomic.colors import element_color, predominant_color
        color = predominant_color(atoms)
        if color is None:
           color = element_color(6)
    else:
        color = color.uint8x4()

    plane_model = PlaneModel(session, name, plane, thickness, radius, color)
    adding_model = find_adding_model(structures)
    if not adding_model:
        session.models.add([plane_model])
    else:
        adding_model.add([plane_model])
    session.logger.info("Plane '%s' placed at %s with normal %s and radius %.1f"
        % (name, plane.origin, plane.normal, radius))
    if show_tool and session.ui.is_gui and not session.in_script:
        from chimerax.core.commands import run
        run(session, "ui tool show Axes/Planes/Centroids", log=False)
    return plane_model

def cmd_define_axis(session, targets=None, *, color=None, radius=None, length=None, name=None, padding=0.0,
        primary=True, secondary=False, tertiary=False, mass_weighting=False, from_point=None, to_point=None,
        per_helix=False, show_tool=True):
    """Wrapper to be called by command line.

       Use chimerax.geometry.vector for other programming applications.
    """
    from chimerax.core.errors import UserError

    from chimerax.core.commands import Axis
    if not (primary or secondary or tertiary):
        raise UserError("One of 'primary', 'secondary', or 'tertiary' must be specified as true.")
    if color is not None:
        color = color.uint8x4()
    from chimerax.atomic import Atoms
    if targets is not None and not isinstance(targets, Atoms):
        # list of planes
        planes = targets
        if mass_weighting:
            raise UserError("Cannot use mass weighting with plane normal determination")
        if per_helix:
            raise UserError("Cannot use 'perHelix' option with plane normal determination")
        if from_point is not None or to_point is not None:
            raise UserError("Cannot specify from/to point with plane normal determination")
        if not planes:
            raise UserError("Must specify a set of atoms or a plane")
        axes = []
        for plane in planes:
            center = plane.plane.origin
            direction = plane.plane.normal
            extent = plane.radius/2 if length is None else length/2
            radius = plane.radius/20 if radius is None else radius
            session.logger.info("Plane normal for %s centered at %s with direction %s, radius %g,"
                " and length %g" % (plane, center, direction, radius, 2*extent))
            axis = AxisModel(session, "normal" if name is None else name, center, direction, extent, radius,
                plane.color if color is None else color, needs_normalization=False)
            plane.add([axis])
            axes.append(axis)
        return axes
    atoms = targets
    axes_info = []
    structure = None
    if atoms is None and from_point is None and to_point is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session, atomic_only=True)
        if not atoms:
            atoms = all_atoms(session, atomic_only=False)
    if from_point is None and to_point is None:
        if not atoms:
            raise UserError("No atoms specified for axis computation")
    elif atoms:
        raise UserError("Cannot specify from/to point *and* a set of atoms for axis")
    min_atoms = 3 if secondary or tertiary else 2
    axis_info = {}
    main_group = None
    if atoms:
        if per_helix:
            from chimerax.atomic import Atom
            main_group = "helix axes" if name is None else name
            # do all helices of the structure, even if specified atoms is less
            for s in atoms.unique_structures:
                for chain in s.chains:
                    c_atoms = chain.existing_residues.atoms
                    backbone = c_atoms.filter(c_atoms.is_backbones(bb_extent=Atom.BBE_MIN))
                    helical = backbone.filter(backbone.residues.is_helices)
                    ss_ids = list(set(helical.residues.ss_ids))
                    ss_ids.sort()
                    for ss_id in ss_ids:
                        helix_atoms = helical.filter(helical.residues.ss_ids == ss_id)
                        if len(helix_atoms) < min_atoms:
                            continue
                        axes_info = determine_axes(helix_atoms, "axis", length, padding, radius,
                            mass_weighting, primary, secondary, tertiary, color)
                        axis_info.setdefault(s, []).append(("%shelix %d" % (("chain %s " % chain.chain_id)
                            if len(s.chains) > 1 else "", ss_id), axes_info))
        else:
            if len(atoms) >= min_atoms:
                adding_model = find_adding_model(atoms.unique_structures)
                axis_info[adding_model] = [(None, determine_axes(atoms, name, length, padding, radius,
                    mass_weighting, primary, secondary, tertiary, color))]

        if not axis_info:
            raise UserError("Must specify %d or more atoms to determine axis/axes" % min_atoms)
    else: # from/toPoint
        if mass_weighting:
            raise UserError("Cannot use mass weighting with from/toPoint axis")
        import numpy
        if from_point is None:
            from_point = numpy.array([0.0, 0.0, 0.0])
        else:
            from_point = numpy.array(from_point)
        if to_point is None:
            to_point = numpy.array([0.0, 0.0, 0.0])
        else:
            to_point = numpy.array(to_point)
        if color is None:
            from chimerax.atomic.colors import element_color
            color = element_color(6)

        center = (to_point + from_point) / 2
        from chimerax.geometry import distance
        extent = distance(from_point, to_point) / 2
        axis_info[None] = [(None, [(name, center, to_point - from_point, extent,
            1.0 if radius is None else radius, color)])]

    from chimerax.core.models import Model
    axes = []
    for structure, s_axes_groups in axis_info.items():
        if main_group:
            overall_grouping_model = Model(main_group, session)
            structure.add([overall_grouping_model])
        elif structure:
            overall_grouping_model = structure
        else:
            overall_grouping_model = session.models

        groupings = {}
        for grouping_name, grp_axes_info in s_axes_groups:
            show_grouping_name = grouping_name is not None
            if len(grp_axes_info) > 1:
                if grouping_name is None:
                    if main_group is None:
                        grouping_name = "axes" if name is None else name
                    else:
                        # 'name' already given to main group
                        grouping_name = "axes"
                add_model = Model(grouping_name, session)
                overall_grouping_model.add([add_model])
            else:
                add_model = overall_grouping_model
            for axis_name, center, direction, extent, radius, color in grp_axes_info:
                if axis_name is None:
                    # per-helix "groups" may contain only one axis or several (secondary, etc.)
                    axis_name = "axis" if grouping_name is None else grouping_name
                    show_grouping_name = False
                elif len(grp_axes_info) == 1 and grouping_name is not None:
                    axis_name = grouping_name
                    show_grouping_name = False
                session.logger.info("Axis '%s%s%s%s' centered at %s with direction %s, radius %g,"
                    " and length %g" % (
                    ("" if structure is None else ("%s/" % structure)),
                    ("" if main_group is None else ("%s/" % main_group)),
                    ("" if not show_grouping_name else ("%s/" % grouping_name)),
                    axis_name, center, direction, radius, 2*extent))
                if structure:
                    inverse = structure.scene_position.inverse()
                    center = inverse * center
                    direction = inverse.zero_translation() * direction

                axis = AxisModel(session, axis_name, center, direction, extent, radius, color,
                    needs_normalization=False)
                axes.append(axis)
                add_model.add([axis])
    if show_tool and session.ui.is_gui:
        from chimerax.core.commands import run
        run(session, "ui tool show Axes/Planes/Centroids", log=False)
    return axes

def find_adding_model(models):
    adding_model = None
    for m in models:
        if adding_model is None:
            adding_model = m
        else:
            models = set()
            cur_model = adding_model
            while cur_model is not None:
                models.add(cur_model)
                cur_model = cur_model.parent
            common_model = m
            while m not in models:
                m = m.parent
                if m is None:
                    return None
            adding_model = m
    return adding_model

def determine_axes(atoms, name, length, padding, radius, mass_weighting, primary, secondary, tertiary,
        color):
    from chimerax.atomic.colors import element_color, predominant_color, average_color
    if color is None:
        color = predominant_color(atoms)
        if color is None:
            # probably rainbowed
            color = average_color(atoms)
    import numpy
    from numpy.linalg import eig, svd, eigh
    if mass_weighting:
        structures = atoms.unique_structures
        classes = set([s.__class__ for s in structures])
        from chimerax.atomic import AtomicStructure
        if len(classes) > 1 and AtomicStructure in classes:
            from chimerax.core.errors import UserError
            raise UserError("Cannot mix markers/centroids and regular atoms when using mass weighting")
        if AtomicStructure in classes:
            weights = atoms.elements.masses
        else:
            weights = atoms.radii
        n = len(atoms)
        mat_weights = weights.reshape((n,1))
        wcoords = mat_weights * atoms.scene_coords
        wsum = weights.sum()
        centroid = wcoords.sum(0) / wsum
        centered = atoms.scene_coords - centroid
        _, vals, vecs = svd(mat_weights * centered, full_matrices=False)
    else:
        centroid = atoms.scene_coords.mean(0)
        centered = atoms.scene_coords - centroid
        _, vals, vecs = svd(centered, full_matrices=False)
    order = reversed(vals.argsort())
    axes_info = []
    for index, axis_name, use in zip(order, ('primary', 'secondary', 'tertiary'),
            (primary, secondary, tertiary)):
        if not use:
            continue
        if int(primary) + int(secondary) + int(tertiary) == 1:
            axis_name = name
        vec = vecs[index]
        if length is None:
            dotted = numpy.dot(centered, vec)
            bounds = (dotted.min(), dotted.max())
            extent = (bounds[1] - bounds[0]) / 2 + padding
            center = ((centroid + bounds[0] * vec) + (centroid + bounds[1] * vec)) / 2
        else:
            extent = length / 2
            center = centroid
        if radius is None:
            if len(centered) > 2:
                # average of distances to axis
                ts = numpy.tensordot(vec, centered, (0, 1)) / numpy.dot(vec, vec)
                line_pts = numpy.outer(ts, vec)
                temp = (centered - line_pts)
                r = numpy.sqrt((temp * temp).sum(-1)).mean(0)
            else:
                r = min(atoms.radii) / 2
        else:
            r = radius
        axes_info.append((axis_name, center, vec, extent, r, color))
    return axes_info

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, FloatArg, ColorArg, PositiveFloatArg
    from chimerax.core.commands import StringArg, Or, Float3Arg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('thickness', PositiveFloatArg), ('padding', FloatArg), ('color', ColorArg),
            ('radius', PositiveFloatArg), ('name', StringArg), ('show_tool', BoolArg)],
        synopsis = 'Create plane'
    )
    register('define plane', desc, cmd_define_plane, logger=logger)

    # atoms have precedence over planes, so only look for planes if no atoms in spec
    class NonEmptyAtomsArg(AtomsArg):
        @classmethod
        def parse(cls, text, session):
            atoms, text, rest = super().parse(text, session)
            if len(atoms) == 0:
                raise AnnotationError("No atoms")
            return atoms, text, rest

    desc = CmdDesc(
        optional=[('targets', Or(NonEmptyAtomsArg, PlaneModelsArg))],
        keyword = [('color', ColorArg), ('radius', PositiveFloatArg), ('length', PositiveFloatArg),
            ('name', StringArg), ('primary', BoolArg), ('secondary', BoolArg), ('tertiary', BoolArg),
            ('mass_weighting', BoolArg), ('from_point', Float3Arg), ('to_point', Float3Arg),
            ('per_helix', BoolArg), ('padding', FloatArg), ('show_tool', BoolArg)],
        synopsis = 'Create axis'
    )
    register('define axis', desc, cmd_define_axis, logger=logger)
