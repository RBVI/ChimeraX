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
        show_tool=None):
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
    results = []
    for traj, traj_atoms in atoms.by_structure:
        from .util import analysis_atoms, analysis_frames
        traj_atoms = analysis_atoms(traj_atoms, exclude_solvent, exclude_hydrogens, exclude_ligands,
            exclude_metals)
        if not traj_atoms:
            raise UserError("No atoms remain after filtering")
        frames = analysis_frames(traj, start, end, step)
        if not frames:
            raise UserError("No frames match start/step/end")
        from .cluster import cluster, ClusterError
        try:
            clusterings = cluster(traj, traj_atoms, frames, status=session.logger.status)
        except ClusterError as e:
            raise UserError(str(e))
        results.append(clusterings)
        if save_file:
            from .cluster import save_clusterings
            save_clusterings(clusterings, save_file)
        if show_tool is not False and session.ui.is_gui and not (session.in_script and show_tool is None):
            from chimerax.std_commands.coordset_gui import CoordinateSetSlider as CSS
            for tool in session.tools:
                if isinstance(tool, CSS) and tool.structure == traj:
                    from .cluster_gui import show_cluster_results
                    show_cluster_results(tool.tool_window, traj, clusterings)
    return results

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
