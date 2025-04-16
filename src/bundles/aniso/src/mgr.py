# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
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

from chimerax.core.triggerset import TriggerSet
triggers = TriggerSet()
triggers.add_trigger("style changed")

def manager_for_structure(session, s, *, create=True):
    for mgr in session.state_managers(_StructureAnisoManager):
        if mgr.structure == s:
            return mgr
    if create:
        return _StructureAnisoManager(session, s)
    return None

from numpy import array
cube_vertices = array([
	(1.0, 1.0, 1.0), (1.0, 1.0, -1.0),
	(1.0, -1.0, 1.0), (1.0, -1.0, -1.0),
	(-1.0, 1.0, 1.0), (-1.0, 1.0, -1.0),
	(-1.0, -1.0, 1.0), (-1.0, -1.0, -1.0)])

# counter-clockwise == normals face out
cube_triangles = array([
	(2,1,0), (3,1,2), (0,1,5), (0,5,4), (0,4,2), (2,4,6),
	(1,3,5), (3,7,5), (3,2,7), (2,6,7), (4,5,7), (4,7,6)])

from chimerax.core.state import StateManager
class _StructureAnisoManager(StateManager):
    def __init__(self, session, structure, *, from_session=False):
        self.init_state_manager(session, "structure thermal ellipsoids")
        self.session = session
        self.structure = structure
        self._cylinder_cache = {}
        if not from_session:
            from chimerax.atomic import Atoms
            self.shown_atoms = Atoms()
            self.atom_depictions = None
            self.drawing_params = {
                'axis_color': None,
                'axis_factor': None,
                'axis_thickness': 0.01,
                'color': None,
                'ellipse_color': None,
                'ellipse_factor': None,
                'ellipse_thickness': 0.02,
                'scale': 1.0,
                'show_ellipsoid': True,
                'smoothing': 3,
                'transparency': None,
            }
            self._create_depictions()
            self._add_handlers()

    def destroy(self):
        for handler in self.handlers:
            handler.remove()
        if not self.structure.deleted:
            self.structure.remove_drawing(self.atom_depictions)
        self.structure = self.session = None
        super().destroy()

    def hide(self, atoms):
        """Hide thermal ellipsoids for these atoms"""
        cur_shown = len(self.shown_atoms)
        self.shown_atoms = self.shown_atoms.subtract(self._filter_aniso(atoms))
        if len(self.shown_atoms) != cur_shown:
            self._create_depictions()

    def show(self, atoms):
        """Show thermal ellipsoids for these atoms"""
        cur_shown = len(self.shown_atoms)
        atoms = self._filter_aniso(atoms)
        self.shown_atoms = self.shown_atoms.merge(atoms)
        if len(self.shown_atoms) == cur_shown:
            return
        self._create_depictions(atoms)

    def style(self, **kw):
        need_rebuild = False
        for param, value in kw.items():
            if self.drawing_params.get(param) != value:
                self.drawing_params[param] = value
                need_rebuild = True

        if need_rebuild:
            self._create_depictions()
            triggers.activate_trigger("style changed", self)

    def _add_handlers(self):
        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [
            self.structure.triggers.add_handler('changes', self._changes_cb),
            self.session.triggers.add_handler(REMOVE_MODELS, self._models_closed_cb)
        ]

    structure_reasons = set(["active_coordset changed", "display changed"])
    atom_reasons = set(["alt_loc changed", "aniso_u changed", "color changed", "coord changed",
        "display changed", "hide changed"])
    def _changes_cb(self, trigger_name, change_info):
        structure, changes = change_info
        if structure.deleted:
            self.destroy()
            return
        if not (set(changes.structure_reasons()).isdisjoint(self.structure_reasons)
        and set(changes.atom_reasons()).isdisjoint(self.atom_reasons)):
            self._create_depictions()
        elif changes.num_deleted_atoms() > 0:
            self._create_depictions()

    def _create_depictions(self, added_atoms=None):
        # added_atoms and self.shown_atoms already filtered for having aniso info
        from chimerax.atomic.shapedrawing import AtomicShapeDrawing
        if not added_atoms:
            if self.atom_depictions is not None:
                self.structure.remove_drawing(self.atom_depictions)
            self.atom_depictions = self.structure.new_drawing('thermal ellipsoid',
                subclass=AtomicShapeDrawing)
            added_atoms = self.shown_atoms

        displayed_atoms = added_atoms.filter(added_atoms.displays)
        explicitly_depicted = set(displayed_atoms.filter(displayed_atoms.hides == 0))
        shapes = []

        from chimerax.atomic import Atoms
        from numpy.linalg import svd
        from numpy import dot, sqrt, negative, cross, array
        dp = self.drawing_params
        drawing_info = []
        for a in explicitly_depicted:
            ignore, lengths, axes = svd(a.aniso_u)
            lengths2 = sqrt(lengths)
            lengths2 *= dp['scale']
            drawing_info.append((Atoms([a]), axes, lengths2))

        smoothing = dp['smoothing']

        from chimerax.surface import calculate_vertex_normals as calc_normals
        if dp['show_ellipsoid']:
            color_param = dp['color']
            transparency = dp['transparency']
            from chimerax.geometry.icosahedron import icosahedron_triangulation
            varray, tarray = icosahedron_triangulation(subdivision_levels=smoothing, sphere_factor=1.0)
            for atoms_arg, axes, lengths2 in drawing_info:
                atom = atoms_arg[0]
                ee = varray * lengths2
                if dot(cross(axes[0], axes[1]), axes[2]) < 0:
                    axes = negative(axes)
                ev = dot(ee, axes)
                ev += atom.coord
                if color_param is None:
                    # match atom
                    color = atom.color
                else:
                    color = color_param
                if transparency is not None:
                    # transparency is a percentage
                    color = color[:-1] + (round((100 - transparency) * 2.55),)
                shapes.append((ev, calc_normals(ev, tarray), tarray, color, atoms_arg, None))

        axis_factor = dp['axis_factor']
        if axis_factor is not None:
            color_param = dp['axis_color']
            thickness = dp['axis_thickness']
            for atoms_arg, axes, lengths2 in drawing_info:
                atom = atoms_arg[0]
                if color_param is None:
                    # match atom
                    color = atom.color
                else:
                    color = color_param
                for axis in range(3):
                    axis_factors = array([thickness]*3)
                    axis_factors[axis] = axis_factor * lengths2[axis]
                    ee = cube_vertices * axis_factors
                    ev = dot(ee, axes)
                    ev += atom.coord
                    tarray = cube_triangles
                    shapes.append((ev, calc_normals(ev, tarray), tarray, color, atoms_arg, None))

        ellipse_factor = dp['ellipse_factor']
        if ellipse_factor is not None:
            color_param = dp['ellipse_color']
            thickness = dp['ellipse_thickness']
            if smoothing not in self._cylinder_cache:
                from chimerax.shape.shape import cylinder_divisions, cylinder_geometry
                nz, nc = cylinder_divisions(1.0, 1.0, 9 * (2**smoothing))
                self._cylinder_cache[smoothing] = cylinder_geometry(1.0, 1.0, nz, nc, True)
            ellipse_vertices, ellipse_triangles = self._cylinder_cache[smoothing]
            for atoms_arg, axes, lengths2 in drawing_info:
                atom = atoms_arg[0]
                if color_param is None:
                    # match atom
                    color = atom.color
                else:
                    color = color_param
                for axis in range(3):
                    verts = ellipse_vertices.copy()
                    if axis < 2:
                        verts[:,axis], verts[:,2] = ellipse_vertices[:,2], ellipse_vertices[:,axis]
                    ellipse_lengths = lengths2 * ellipse_factor
                    ellipse_lengths[axis] = thickness
                    ee = verts * ellipse_lengths
                    ev = dot(ee, axes)
                    ev += atom.coord
                    tarray = ellipse_triangles
                    shapes.append((ev, calc_normals(ev, tarray), tarray, color, atoms_arg, None))
        if shapes:
            self.atom_depictions.add_shapes(shapes)

    def _filter_aniso(self, atoms):
        return atoms[atoms.has_aniso_u]

    def _models_closed_cb(self, trigger_name, closed_models):
        if self.structure in closed_models:
            self.destroy()
            return

    def report_style_settings(self):
        from chimerax.core.logger import html_table_params
        table_lines = [
            '<table %s>' % html_table_params,
            '  <thead>',
            '    <tr>',
            '      <th colspan="2">Thermal ellipsoid style settings for %s</th>' % self.structure,
            '    </tr>',
            '    <tr>',
            '      <th>Parameter</th>',
            '      <th>Value</th>',
            '    </tr>',
            '  </thead>',
            '  <tbody>',
        ]
        from chimerax.core.commands import camel_case
        param_names = sorted(list(self.drawing_params.keys()))
        for param in param_names:
            value = self.drawing_params[param]
            if value is None:
                if param.endswith("_color"):
                    val_text = "from atom"
                elif param.endswith("_factor"):
                    val_text = "off"
                else:
                    # transparency
                    val_text = "none"
            elif isinstance(value, bool):
                val_text = repr(value).lower()
            elif isinstance(value, (int, float)):
                val_text = "%g" % value
            else:
                # color
                from chimerax.core.colors import color_name
                val_text = color_name(value)
            table_lines.extend([
                '    <tr>',
                '      <td style="text-align:center">%s</td>' % camel_case(param),
                '      <td style="text-align:center">%s</td>' % val_text,
                '    </tr>',
            ])
        table_lines.extend(['  </tbody>', '</table>'])

        self.session.logger.info('\n'.join(table_lines), is_html=True)

    def reset_state(self, session):
        self.destroy()

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = cls(session, data['structure'], from_session=True)
        inst.atom_depictions = data['atom_depictions']
        inst.structure.add_drawing(inst.atom_depictions)
        inst.drawing_params = data['drawing_params']
        inst.shown_atoms = data['shown_atoms']
        def delayed_registration(*args, inst=inst):
            inst._add_handlers()
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER
        from chimerax.atomic import get_triggers
        get_triggers().add_handler('changes done', delayed_registration)
        return inst

    def take_snapshot(self, session, flags):
        data = {
            'atom_depictions': self.atom_depictions,
            'drawing_params': self.drawing_params,
            'shown_atoms': self.shown_atoms,
            'structure': self.structure,
        }
        return data
