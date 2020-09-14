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
        if not isinstance(obj, PlaneModel):
            return NotImplemented
        from chimerax.geometry import angle
        degrees = angle(self.position.transform_vector(self.plane.normal),
            obj.position.transform_vector(obj.plane.normal))
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


def cmd_define_plane(session, atoms, *, thickness=defaults["plane_thickness"], padding=0.0, color=None,
        radius=None, name="plane"):
    """Wrapper to be called by command line.

       Use chimerax.axes_planes.plane for other programming applications.
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
           color = element_color(a.element.number)
    else:
        color = color.uint8x4()

    plane_model = PlaneModel(session, name, plane, thickness, radius+padding, color)
    if len(structures) > 1:
        session.models.add([plane_model])
    else:
        structures[0].add([plane_model])
    session.logger.info("Plane %s' placed at %s with normal %s" % (name, plane.origin, plane.normal))
    return plane_model


def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, FloatArg, ColorArg, PositiveFloatArg
    from chimerax.core.commands import StringArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('thickness', PositiveFloatArg), ('padding', FloatArg), ('color', ColorArg),
            ('radius', PositiveFloatArg), ('name', StringArg)],
        synopsis = 'Create plane'
    )
    register('define plane', desc, cmd_define_plane, logger=logger)
