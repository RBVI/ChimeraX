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

from chimerax.core.errors import UserError
from chimerax.add_charge import ChargeMethodArg

def cmd_coulombic(session, atoms, *, surfaces=None, his_scheme=None, offset=1.4, spacing=1.0,
        padding=5.0, map=False, palette=None, range=None, dist_dep=True, dielectric=4.0,
        charge_method=ChargeMethodArg.default_value, key=False):
    if map:
        session.logger.warning("Computing electrostatic volume map not yet supported")
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
    for surf_atoms, shown_atoms, srf in atoms_per_surf:
        if srf is None:
            session.logger.status("Creating surface", secondary=True)
            from chimerax.surface import surface
            data = [(surf.atoms, surf)
                for surf in surface(session, surf_atoms if shown_atoms is None else shown_atoms)]
        else:
            data = [(surf_atoms, srf)]
        session.logger.status("Computing electrostatics", secondary=True)
        for charged_atoms, target_surface in data:
            undo_owners.append(target_surface)
            undo_old_vals.append(target_surface.vertex_colors)
            if target_surface.normals is None:
                session.logger.warning("Surface %s has no vertex normals set, using distance from surface"
                    " of 0 instead of %g" % (target_surface, offset))
                target_points = target_surface.vertices
            else:
                target_points = target_surface.vertices + offset * target_surface.normals
            import numpy, os
            # Make sure _esp can runtime link shared library libarrays.
            from chimerax import arrays ; arrays.load_libarrays()
            from ._esp import potential_at_points
            cpu_count = os.cpu_count()
            vertex_values = potential_at_points(
                target_surface.scene_position.transform_points(target_points), charged_atoms.scene_coords,
                numpy.array([a.charge for a in charged_atoms], dtype=numpy.double), dist_dep, dielectric,
                1 if cpu_count is None else cpu_count)
            rgba = cmap.interpolated_rgba(vertex_values)
            from numpy import uint8, amin, mean, amax
            rgba8 = (255*rgba).astype(uint8)
            target_surface.vertex_colors = rgba8
            undo_new_vals.append(rgba8)
            session.logger.info("Coulombic values for %s: minimum, %.2f, mean %.2f, maximum %.2f"
                % (target_surface, amin(vertex_values), mean(vertex_values), amax(vertex_values)))
    undo_state.add(undo_owners, "vertex_colors", undo_old_vals, undo_new_vals, option="S")
    session.undo.register(undo_state)
    if key:
        from chimerax.color_key import show_key
        show_key(session, cmap)

    session.logger.status("", secondary=True)
    session.logger.status("Finished computing Coulombic potential%s" % (" map" if map else ""))

"""
def coulombic_map(session, charged_atoms, target_surface, offset, spacing, padding, vol_name):
    data, bounds = calculate_map(target_surface, charged_atoms, spacing, offset + padding)
    #TODO
"""

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, SurfacesArg, EnumOf, FloatArg
    from chimerax.core.commands import BoolArg, ColormapArg, ColormapRangeArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('atoms', Or(AtomsArg, EmptyArg))],
        keyword = [
            ('surfaces', SurfacesArg),
            ('his_scheme', EnumOf(['HIP', 'HIE', 'HID'])),
            ('offset', FloatArg),
            ('spacing', FloatArg),
            ('padding', FloatArg),
            ('map', BoolArg),
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
