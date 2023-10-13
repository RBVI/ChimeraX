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

from chimerax.core.errors import UserError
from chimerax.add_charge import ChargeMethodArg

def cmd_coulombic(session, atoms, *, surfaces=None, his_scheme=None, offset=1.4, gspacing=None,
        gpadding=None, map=None, palette=None, range=None, dist_dep=True, dielectric=4.0,
        charge_method=ChargeMethodArg.default_value, key=False):
    session.logger.status("Computing Coulombic potential%s" % (" map" if map else ""))
    if palette is None:
        from chimerax.core.colors import BuiltinColormaps
        cmap = BuiltinColormaps["red-white-blue"]
    else:
        cmap = palette
    if not cmap.values_specified:
        rmin, rmax = (-10.0, 10.0) if range is None else range
        cmap = cmap.linear_range(rmin, rmax)
    session.logger.status("Matching atoms to surfaces", secondary=True)
    atoms_per_surf = []
    from chimerax.atomic import all_atomic_structures, MolecularSurface, all_atoms
    if atoms is None:
        if surfaces is None:
            # surface all chains
            for struct in all_atomic_structures(session):
                # don't create surface until charges checked
                if struct.num_chains == 0:
                    atoms_per_surf.append((struct.atoms, None, None))
                else:
                    for chain in struct.chains:
                        atoms_per_surf.append((chain.existing_residues.atoms, None, None))
        else:
            for srf in surfaces:
                if isinstance(srf, MolecularSurface):
                    atoms_per_surf.append((srf.atoms, None, srf))
                else:
                    atoms_per_surf.append((all_atoms(session), None, srf))
    else:
        if surfaces is None:
            # on a per-structure basis, determine if the atoms contain any polymers, and if so then
            # surface those chains (and not non-polymers); otherwise surface the atoms
            by_chain = {}
            for struct, chain_id, chain_atoms in atoms.by_chain:
                chains = chain_atoms.unique_residues.unique_chains
                if chains:
                    by_chain.setdefault(struct, {})[chains[0]] = chain_atoms
            for struct, struct_atoms in atoms.by_structure:
                try:
                    for chain, shown_atoms in by_chain[struct].items():
                        chain_atoms = chain.existing_residues.atoms
                        atoms_per_surf.append((chain_atoms, chain_atoms & shown_atoms, None))
                except KeyError:
                    atoms_per_surf.append((struct_atoms, None, None))
        else:
            for srf in surfaces:
                atoms_per_surf.append((atoms, None, srf))

    # check whether the atoms have charges, and if not, that we know how to assign charges
    # to the requested atoms
    needs_assignment = set()
    for surf_atoms, shown_atoms, srf in atoms_per_surf:
        for r in surf_atoms.unique_residues:
            if getattr(r, '_coulombic_his_scheme', his_scheme) != his_scheme:
                # should only be set on HIS residues
                needs_assignment.add(r)
            else:
                for a in r.atoms:
                    try:
                        a.charge + 1.0
                    except (AttributeError, TypeError):
                        needs_assignment.add(r)
                        break
    if needs_assignment:
        session.logger.status("Assigning charges", secondary=True)
        from .coulombic import assign_charges
        from chimerax.add_charge import ChargeError
        try:
            assign_charges(session, needs_assignment, his_scheme, charge_method,
                status=session.logger.status)
        except ChargeError as e:
            session.logger.status("")
            raise UserError(str(e))

    # Since electrostatics are long range, unlike mlp, don't compute a map (with a distance cutoff)
    # by default.  Instead, compute the values at the surface vertices directly.  Only compute a
    # map afterward if requested.
    from chimerax.core.undo import UndoState
    undo_state = UndoState('coulombic')
    undo_owners = []
    undo_old_vals = []
    undo_new_vals = []
    grid_data = []
    for surf_atoms, shown_atoms, srf in atoms_per_surf:
        if srf is None:
            session.logger.status("Creating surface", secondary=True)
            from chimerax.surface import surface
            data = [(surf.atoms, surf)
                for surf in surface(session, surf_atoms if shown_atoms is None else shown_atoms)]
        else:
            data = [(surf_atoms, srf)]
        grid_data.extend(data)
        session.logger.status("Computing electrostatics", secondary=True)
        for charged_atoms, target_surface in data:
            for clip_surface in [cs for cs in target_surface.child_models()
                    if getattr(cs, 'is_clip_cap', False)]:
                clip_surface.auto_recolor_vertices = lambda *args, ses=session, s=clip_surface, \
                    charged_atoms=charged_atoms, dist_dep=dist_dep, dielectric=dielectric, \
                    cmap=cmap, f=color_vertices: f(ses, s, 0.0, charged_atoms, dist_dep, dielectric,
                    cmap, log=False)
            color_vertices(session, target_surface, offset, charged_atoms, dist_dep, dielectric, cmap,
                undo_info=(undo_owners, undo_old_vals, undo_new_vals))
    undo_state.add(undo_owners, "vertex_colors", undo_old_vals, undo_new_vals, option="S")
    session.undo.register(undo_state)
    if key:
        from chimerax.color_key import show_key
        show_key(session, cmap)

    session.logger.status("", secondary=True)
    session.logger.status("Finished coloring surfaces")
    if map is None:
        map = gspacing is not None or gpadding is not None
    if not map:
        return
    session.logger.status("Computing Coulombic potential grids")
    if gspacing is None:
        gspacing = 1.0
    if gpadding is None:
        gpadding = 5.0
    import numpy, os
    import chimerax.arrays # Make sure _esp can runtime link shared library libarrays.
    from ._esp import potential_at_points
    from chimerax.map import volume_from_grid_data
    from chimerax.map_data import ArrayGridData
    cpu_count = os.cpu_count()
    for atoms, surf in grid_data:
        coords = atoms.coords
        min_xyz = numpy.min(coords, axis=0) - [gpadding+gspacing/2.0]*3
        max_xyz = numpy.max(coords, axis=0) + [gpadding+gspacing/2.0]*3
        x_range = numpy.arange(min_xyz[0], max_xyz[0], gspacing)
        y_range = numpy.arange(min_xyz[1], max_xyz[1], gspacing)
        z_range = numpy.arange(min_xyz[2], max_xyz[2], gspacing)
        grid_vertices = numpy.array([(x,y,z) for x in x_range for y in y_range for z in z_range])
        grid_potentials = potential_at_points(grid_vertices,
            atoms.coords, numpy.array([a.charge for a in atoms], dtype=numpy.double),
            dist_dep, dielectric, 1 if cpu_count is None else cpu_count)
        grid_potentials.shape = (len(x_range), len(y_range), len(z_range))
        agd = ArrayGridData(grid_potentials.transpose(), min_xyz, [gspacing]*3)
        agd.polar_values = True
        v = volume_from_grid_data(agd, session, open_model=False)
        v.set_parameters(surface_levels=[-10, 10],
            surface_colors=[(1.0, 0.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5)], cap_faces=False)
        v.update_drawings() # for surfaces to show so that volume viewer doesn't override them
        v.name = "coulombic grid for " + surf.name
        surf.parent.add([v])
        surf.display = False
    session.logger.status("Finished computing Coulombic potential grids")

def color_vertices(session, surface, offset, charged_atoms, dist_dep, dielectric, cmap, *, log=True,
        undo_info=None):
    if surface.vertices is None:
        return
    if undo_info:
        undo_owners, undo_old_vals, undo_new_vals = undo_info
        undo_owners.append(surface)
        undo_old_vals.append(surface.vertex_colors)
    if surface.normals is None:
        if log:
            session.logger.warning("Surface %s has no vertex normals set, using distance from surface of 0"
                " instead of %g" % (surface, offset))
        target_points = surface.vertices
    else:
        target_points = surface.vertices + offset * surface.normals
    arv = surface.auto_recolor_vertices
    import numpy, os
    import chimerax.arrays # Make sure _esp can runtime link shared library libarrays.
    from ._esp import potential_at_points
    cpu_count = os.cpu_count()
    vertex_values = potential_at_points(surface.scene_position.transform_points(target_points),
        charged_atoms.scene_coords, numpy.array([a.charge for a in charged_atoms], dtype=numpy.double),
        dist_dep, dielectric, 1 if cpu_count is None else cpu_count)
    rgba = cmap.interpolated_rgba(vertex_values)
    from numpy import uint8, amin, mean, amax
    rgba8 = (255*rgba).astype(uint8)
    surface.vertex_colors = rgba8
    if undo_info:
        undo_new_vals.append(rgba8)
    surface.auto_recolor_vertices = arv
    if log:
        session.logger.info("Coulombic values for %s: minimum, %.2f, mean %.2f, maximum %.2f"
            % (surface, amin(vertex_values), mean(vertex_values), amax(vertex_values)))

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, SurfacesArg, EnumOf, FloatArg
    from chimerax.core.commands import BoolArg, ColormapArg, ColormapRangeArg, StringArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('atoms', Or(AtomsArg, EmptyArg))],
        keyword = [
            ('surfaces', SurfacesArg),
            ('gpadding', FloatArg),
            ('map', BoolArg),
            ('gspacing', FloatArg),
            ('his_scheme', EnumOf(['HIP', 'HIE', 'HID'])),
            ('offset', FloatArg),
            ('palette', ColormapArg),
            ('range', ColormapRangeArg),
            ('dist_dep', BoolArg),
            ('dielectric', FloatArg),
            ('charge_method', ChargeMethodArg),
            ('key', BoolArg),
        ],
        synopsis = 'Color surfaces by coulombic potential'
    )
    register("coulombic", desc, cmd_coulombic, logger=logger)
