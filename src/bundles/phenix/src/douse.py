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
def phenix_douse(session, map, near_model, keep_input_water = True, phenix_location = None,
                 residue_range = 5, map_range = 8, verbose = False):

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
    map_0, shift = _fix_map_origin(map)

    # Save model to file.
    from chimerax.pdb import save_pdb
    save_pdb(session, path.join(temp_dir,'model.pdb'),
             models = [near_model], rel_model = map_0)

    # Run phenix.douse
    output_model = _run_douse(session, exe_path, 'map.mrc', 'model.pdb', temp_dir,
                              keep_input_water = keep_input_water, verbose = verbose)
    output_model.name = near_model.name + ' douse'
    output_model.position = map.scene_position
    if shift is not None:
        output_model.atoms.coords += shift
    session.models.add([output_model])

    # Report predicted waters and input waters
    msg, nwaters = _describe_new_waters(near_model, output_model, keep_input_water, map.name)
    session.logger.info(msg, is_html=True)

    # Show only waters and nearby residues and transparent map near waters.
    if nwaters > 0:
        _show_waters(near_model, output_model, residue_range, map, map_range)
    
    return output_model

# ---------------------------------------------------------------------------------------
#
def _fix_map_origin(map):
    '''
    Douse ignores the MRC file origin so if it is non-zero take the
    atom coordinates relative to the map assuming zero origin.
    '''
    if tuple(map.data.origin) != (0,0,0):
        from chimerax.geometry import translation
        shift = map.data.origin
        from chimerax.core.models import Model
        map_0 = Model('douse shift coords', map.session)
        map_0.position = map.scene_position * translation(shift)
    else:
        shift = None
        map_0 = map
    return map_0, shift

# ---------------------------------------------------------------------------------------
#
def _run_douse(session, exe_path, map_path, model_path, temp_dir,
               keep_input_water = True, verbose = False):
    '''
    Run douse in a subprocess and return the model with predicted waters.
    '''
    args = [exe_path, map_path, model_path]
    if keep_input_water:
        args.append('keep_input_water=true')
    session.logger.status(f'Running {exe_path} in directory {temp_dir}')
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
    from os import path
    output_path = path.join(temp_dir,'douse_000.pdb')
    from chimerax.pdb import open_pdb
    models, info = open_pdb(session, output_path, log_info = False)

    return models[0]

# ---------------------------------------------------------------------------------------
#
def _describe_new_waters(input_model, output_model, keep_input_water, map_name):
    input_wat_res, new_wat_res, dup_wat_res, dup_input_wat_res = \
        _compare_waters(input_model, output_model)
      
    ninput = len(input_wat_res)
    nindup = len(dup_input_wat_res)
    noutput = len(new_wat_res) + len(dup_wat_res)
    nnew = len(new_wat_res)
    ndup = len(dup_wat_res)
    sel_out = f'select #{output_model.id_string}:HOH'
    new_res_spec = _residue_specifier(output_model, new_wat_res)
    water_spec = f'#{output_model.id_string}:HOH'
    sel_new = f'select {new_res_spec} & {water_spec}'
    dup_res_spec = _residue_specifier(output_model, dup_wat_res)
    sel_dup = f'select {dup_res_spec} & {water_spec}'
    xtra_res_spec = _residue_specifier(input_model, (input_wat_res - dup_input_wat_res))
    sel_xtra = f'select {xtra_res_spec}'

    if keep_input_water:
        msg = (
            f'Placed <a href="cxcmd:{sel_new}">{nnew} new</a> waters'
            f' in map "{map_name}" near model "{input_model.name}"')
    else:
        msg = (
            f'Placed <a href="cxcmd:{sel_out}">{noutput} waters</a>'
            f' in map "{map_name}" near model "{input_model.name}"<br>'
            f' <a href="cxcmd:{sel_new}">{nnew} new</a> waters,'
            f' <a href="cxcmd:{sel_dup}">{ndup} matching</a> input waters,'
            f' <a href="cxcmd:{sel_xtra}">{ninput-nindup} input waters not found</a>')

    return msg, nnew

# ---------------------------------------------------------------------------------------
#
def _compare_waters(input_model, output_model, overlap_distance = 2):
    '''
    Find how many output waters overlap input waters.
    '''
    # Get water residues in input and output models
    ires = input_model.residues
    input_waters = ires[ires.names == 'HOH']
    ninput = len(input_waters)
    ores = output_model.residues
    output_waters = ores[ores.names == 'HOH']
    noutput = len(output_waters)

    # Get water oxygen coodinates and see which overlap.
    from chimerax.atomic import Atoms
    ia = Atoms([r.find_atom('O') for r in input_waters])
    ixyz = ia.scene_coords
    oa = Atoms([r.find_atom('O') for r in output_waters])
    oxyz = oa.scene_coords
    from chimerax.geometry import find_close_points
    ii,io = find_close_points(ixyz, oxyz, overlap_distance)
    dup_wat_res = output_waters[io]	# Output water residues near input water residues
    new_wat_res = output_waters - dup_wat_res	# Output waters not near input waters
    dup_input_wat_res = input_waters[ii]	# Input waters near output waters

    return input_waters, new_wat_res, dup_wat_res, dup_input_wat_res

# ---------------------------------------------------------------------------------------
#
def _residue_specifier(model, residues):
    res_ids = ','.join('%d' % r.number for r in residues)
    return f'#{model.id_string}:{res_ids}'

# ---------------------------------------------------------------------------------------
#
def _show_waters(input_model, output_model, residue_range, map, map_range):
    m_id = output_model.id_string
    commands = [f'hide #{m_id} atoms,ribbons',
                f'show #{m_id}:HOH',
                f'transparency #{map.id_string} 50',
                f'hide #{input_model.id_string} model']
    if residue_range > 0:
        commands.append(f'show #{m_id}:HOH :< {residue_range}')
    if map_range > 0:
        commands.append(f'volume zone #{map.id_string} near #{m_id}:HOH range {map_range}')
    cmd = ' ; '.join(commands)
    from chimerax.core.commands import run
    run(output_model.session, cmd, log = False)

# ---------------------------------------------------------------------------------------
#
def register_phenix_douse_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFolderNameArg, BoolArg, FloatArg
    from chimerax.map import MapArg
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        required = [('map', MapArg)],
        keyword = [('near_model', AtomicStructureArg),
                   ('keep_input_water', BoolArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('residue_range', FloatArg),
                   ('map_range', FloatArg),
                   ('verbose', BoolArg),
        ],
        required_arguments = ['near_model'],
        synopsis = 'Place water molecules in map'
    )
    register('phenix douse', desc, phenix_douse, logger=logger)
