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

def cmd_cluster(session, atoms=None, *, start=None, step=None, end=None, exclude_solvent=True,
        exclude_hydrogens=True, exclude_ligands=False, exclude_metals="alkali", save_file=None,
        show_tool=True):
    '''
    Cluster trajectory frames
    '''
    if atoms is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)
    atoms = atoms.filter(atoms.structures.num_coordsets > 1)
    from chimerax.core.errors import UserError
    if not atoms:
        raise UserError("No trajectory atoms specified")
    for traj, traj_atoms in atoms.by_structure:
        from numpy import logical_not, logical_and
        if exclude_solvent:
            traj_atoms = traj_atoms.filter(traj_atoms.structure_categories != "solvent")
            ions = traj_atoms.structure_categories == "ions"
            metals = traj_atoms.elements.is_metal
            non_metal_ions = logical_and(ions, logical_not(metals))
            traj_atoms = traj_atoms.filter(logical_not(non_metal_ions))
        if exclude_hydrogens:
            traj_atoms = traj_atoms.filter(traj_atoms.elements.numbers > 1)
        if exclude_ligands:
            traj_atoms = traj_atoms.filter(traj_atoms.structure_categories != "ligand")
        if exclude_metals:
            if exclude_metals == "alkali":
                metals = traj_atoms.elements.is_alkali_metal
            else:
                metals = traj_atoms.elements.is_metal
            traj_atoms = traj_atoms.filter(logical_not(metals))
        if not traj_atoms:
            raise UserError("No atoms remain after filtering")
        all_cs_ids = set(traj.coordset_ids)
        if start is None:
            start = min(all_cs_ids)
        if end is None:
            end = max(all_cs_ids)
        if step is None:
            step = 1 + int(len(all_cs_ids)/300)
        frames = []
        for fn in range(start, end+1, step):
            if fn in all_cs_ids:
                frames.append(fn)
        if not frames:
            raise UserError("No frames match start/step/end")
        from .cluster import cluster, ClusterError
        try:
            clusterings = cluster(traj, traj_atoms, frames, status=session.logger.status)
        except ClusterError as e:
            raise UserError(str(e))
        if save_file:
            from .cluster import save_clusterings
            save_clusterings(clusterings, save_file)
        if show_tool and session.ui.is_gui:
            print("Show tool now")

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg
    from chimerax.core.commands import IntArg, PositiveIntArg, EnumOf
    from chimerax.core.commands import Or, EmptyArg, SaveFileNameArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('atoms', Or(AtomsArg, EmptyArg))],
        keyword = [
            ('start', IntArg), ('step', PositiveIntArg), ('end', IntArg),
            ('exclude_solvent', BoolArg), ('exclude_hydrogens', BoolArg), ('exclude_ligands', BoolArg),
            ('exclude_metals', Or(BoolArg, EnumOf(["alkali"]))),
            ('save_file', SaveFileNameArg), ('show_tool', BoolArg),
        ],
        synopsis = 'Cluster trajectory frames'
    )
    register('md cluster', desc, cmd_cluster, logger=logger)
