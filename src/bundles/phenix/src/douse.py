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

# ---------------------------------------------------------------------------------------
# Command and tool to place waters in cryoEM maps using Phenix douse.
#
def phenix_douse(session, map, near_model, phenix_location = None, verbose = False):

    # Find the phenix.douse executable
    from .locate import _find_phenix_command
    exe_path = _find_phenix_command(session, 'phenix.douse', phenix_location)

    # Setup temporary directory to run phenix.douse.
    from tempfile import TemporaryDirectory
    d = TemporaryDirectory(prefix = 'phenix_douse_')  # Will be cleaned up when object deleted.
    temp_dir = d.name

    # Save map to file
    from os import path
    from chimerax.map_data import save_grid_data
    save_grid_data([map.data], path.join(temp_dir,'map.mrc'), session)

    # Douse ignores the MRC file origin so if it is non-zero
    # shift the atom coordinates so they align with the origin 0 map.
    if tuple(map.data.origin) != (0,0,0):
        from chimerax.geometry import translation
        shift = map.data.origin
        from chimerax.core.models import Model
        map_0 = Model('douse shift coords', session)
        map_0.position = map.scene_position * translation(shift)
    else:
        shift = None
        map_0 = map

    # Save model to file.
    from chimerax.pdb import save_pdb
    save_pdb(session, path.join(temp_dir,'model.pdb'),
             models = [near_model], rel_model = map_0)

    # Run phenix.douse
    args = [exe_path, 'map.mrc', 'model.pdb']
    session.logger.status(f'Running {exe_path} in directory {temp_dir}', log = True)
    import subprocess
    p = subprocess.run(args, capture_output = True, cwd = temp_dir)
    if p.returncode != 0:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = (f'phenix.douse exited with error code {p.returncode}\n\n' +
               f'Command: {cmd}\n\n' +
               f'stdout:\n{out}\n\n' +
               f'stderr:\n{err}')
        from chimerax.core.errors import UserError
        raise UserError(msg)

    # Log phenix douse command output
    if verbose:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = f'<pre><b>Command</b>:\n\n{cmd}\n\n<b>stdout</b>:\n\n{out}'
        if err:
            msg += f'\n\n<b>stderr</b>:\n\n{err}'
        msg += '</pre>'
        session.logger.info(msg, is_html = True)

    # Open new model with added waters
    from chimerax.pdb import open_pdb
    models, info = open_pdb(session, path.join(temp_dir,'douse_000.pdb'), log_info = False)
    m = models[0]
    m.name = near_model.name + ' douse'
    m.position = map.scene_position
    if shift is not None:
        m.atoms.coords += shift
    session.models.add(models)
    
    # Report number of waters added
    res = m.residues
    waters = res[res.names == 'HOH']
    nwat = len(waters)
    session.logger.info(f'Found {nwat} waters in map {map.name} near model {near_model.name}')
    
    # Show only waters and nearby residues and transparent map near waters.
    if nwat > 0:
        m_id = m.id_string
        res_range = 5
        map_range = 8
        commands = [f'hide #{m_id} atoms,ribbons',
                    f'show #{m_id}:HOH :< {res_range}',
                    f'volume zone #{map.id_string} near #{m_id}:HOH range {map_range}',
                    f'transparency #{map.id_string} 50',
                    f'hide #{near_model.id_string} model']
        cmd = ' ; '.join(commands)
        from chimerax.core.commands import run
        run(session, cmd, log = False)
    
    return m
    
# ---------------------------------------------------------------------------------------
#
def register_phenix_douse_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFolderNameArg, BoolArg
    from chimerax.map import MapArg
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        required = [('map', MapArg)],
        keyword = [('near_model', AtomicStructureArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('verbose', BoolArg),
        ],
        required_arguments = ['near_model'],
        synopsis = 'Place water molecules in map'
    )
    register('phenix douse', desc, phenix_douse, logger=logger)
