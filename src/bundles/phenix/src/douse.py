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

#
# Command to place waters in cryoEM maps using Phenix douse.
#
from chimerax.core.tasks import Job
from time import time

class DouseJob(Job):

    SESSION_SAVE = False

    def __init__(self, session, executable_location, optional_args, map_file_name, model_file_name,
            positional_args, temp_dir, keep_input_water, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, map_file_name, model_file_name,
            positional_args, temp_dir, keep_input_water, verbose, callback, blocking=block)

    def run(self, session, executable_location, optional_args, map_file_name, model_file_name,
            positional_args, temp_dir, keep_input_water, verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                results = _run_douse_subprocess(session, executable_location, optional_args, map_file_name,
                    model_file_name, positional_args, temp_dir, keep_input_water, verbose)
            finally:
                self._running = False
            self.session.ui.thread_safe(callback, results)
        import threading
        thread = threading.Thread(target=threaded_run, daemon=True)
        thread.start()
        super().run()

    def monitor(self):
        delta = int(time() - self.start_t + 0.5)
        if delta < 60:
            time_info = "%d seconds" % delta
        elif delta < 3600:
            minutes = delta // 60
            seconds = delta % 60
            time_info = "%d minutes and %d seconds" % (minutes, seconds)
        else:
            hours = delta // 3600
            minutes = (delta % 3600) // 60
            seconds = delta % 60
            time_info = "%d:%02d:%02d" % (hours, minutes, seconds)
        ses = self.session
        ses.ui.thread_safe(ses.logger.status, "Douse job still running (%s)" % time_info)

    def next_check(self):
        self._monitor_time += self._monitor_interval
        return self._monitor_time

    def running(self):
        return self._running


def phenix_douse(session, map, near_model, *, block=None, far_water=False, keep_input_water=True,
        map_range=8, phenix_location=None, residue_range=5, verbose=False, option_arg=[], position_arg=[]):

    # Find the phenix.douse executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.douse', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

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
    douse_keep_input_water = (keep_input_water and far_water)
    # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
    # douse runs
    callback = lambda douse_model, *args, session=session, shift=shift, near_model=near_model, \
        keep_input_water=keep_input_water, far_water=far_water, map=map, residue_range=residue_range, \
        map_range=map_range, d_ref=d: _process_results(session, douse_model, shift, near_model,
        keep_input_water, far_water, map, residue_range, map_range)
    DouseJob(session, exe_path, option_arg, "map.mrc", "model.pdb", position_arg, temp_dir,
        douse_keep_input_water, verbose, callback, block)


def _process_results(session, douse_model, shift, near_model, keep_input_water, far_water, map,
        residue_range, map_range):
    douse_model.position = map.scene_position
    if shift is not None:
        douse_model.atoms.coords += shift

    # Copy new waters
    model, msg, nwaters, compared_waters = _copy_new_waters(douse_model, near_model, keep_input_water,
        far_water, map.name)
    douse_model.delete()

    # Report predicted waters and input waters
    session.logger.info(msg, is_html=True)

    # Show only waters and nearby residues and transparent map near waters.
    if nwaters > 0:
        _show_waters(near_model, model, residue_range, map, map_range)
        if session.ui.is_gui:
            from .tool import DouseResultsViewer
            DouseResultsViewer(session, "Douse Results", near_model, model, compared_waters)


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


def _run_douse_subprocess(session, exe_path, optional_args, map_filename, model_filename, positional_args,
        temp_dir, keep_input_water, verbose):
    '''
    Run douse in a subprocess and return the model with predicted waters.
    '''
    args = [exe_path] + optional_args + [map_filename, model_filename] + positional_args
    if keep_input_water:
        args.append('keep_input_water=true')
    tsafe=session.ui.thread_safe
    logger = session.logger
    tsafe(logger.status, f'Running {exe_path} in directory {temp_dir}')
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
        tsafe(logger.info, msg, is_html=True)

    # Open new model with added waters
    from os import path
    output_path = path.join(temp_dir,'douse_000.pdb')
    from chimerax.pdb import open_pdb
    models, info = open_pdb(session, output_path, log_info = False)

    return models[0]


#NOTE: We don't use a REST server; code retained for reference
"""
def _run_douse_rest_server(session, rest_server, map_filename, model_filename, temp_dir,
                           keep_input_water = True, verbose = False):
    '''
    Run douse using the Phenix REST server and return the model with predicted waters.
    '''
    from os import path
    model_path = path.join(temp_dir, model_filename)
    map_path = path.join(temp_dir, map_filename)
    args = [model_path, map_path]
    if keep_input_water:
        args.append('keep_input_water=true')

    # Run job
    session.logger.status(f'Running douse in directory {temp_dir}')
    job = rest_server.start_job('douse', args)
    job.wait()
    result = job.result

    # Check for error.
    if result is None:
        cmd = ' '.join(args)
        msg = (f'phenix.douse exited with an error\n\n' +
               f'Command: {cmd}\n\n' +
               f'stdout:\n{job.stdout}\n\n' +
               f'stderr:\n{job.stderr}')
        from chimerax.core.errors import UserError
        raise UserError(msg)

    # Log phenix douse command output
    if verbose:
        cmd = ' '.join(args)
        msg = f'<pre><b>Command</b>:\n\n{cmd}\n\n<b>stdout</b>:\n\n{job.stdout}'
        if job.stderr:
            msg += f'\n\n<b>stderr</b>:\n\n{job.stderr}'
        msg += '</pre>'
        session.logger.info(msg, is_html = True)

    # Open new model with added waters
    douse_pdb = result['output_file']
    from chimerax.pdb import open_pdb
    models, info = open_pdb(session, douse_pdb, log_info = False)

    return models[0]
"""


def _copy_new_waters(douse_model, near_model, keep_input_water, far_water, map_name):
    model = near_model.copy()
    model.position = douse_model.position
    model.name = near_model.name + ' douse'

    # Add found water molecules to copy of input molecule.
    if keep_input_water:
        input_waters, douse_only_waters, douse_both_waters, input_both_waters = \
            compared_waters = _compare_waters(near_model, douse_model)
    else:
        douse_waters = _water_residues(douse_model)
        _water_residues(model).delete()
        compared_waters = None
    added_wat_res = _add_waters(model, douse_only_waters)
    if compared_waters:
        compared_waters = compared_waters[:1] + (added_wat_res,) + compared_waters[2:]

    model.session.models.add([model])	# Need to assign id number for use in log message

    # Create log message describing found waters with links
    # to select them.
    sel_new, nnew = _select_command(model, added_wat_res)
    long_message = keep_input_water and len(input_waters) > 0 and not far_water
    msg = (f'Placed <a href="cxcmd:{sel_new}">{nnew}%s waters</a>'
           f' in map "{map_name}" near model "{near_model.name}"') % (" new" if long_message else "")
    if long_message:
        sel_dup, ndup = _select_command(model, input_both_waters)
        sel_xtra, nxtra = _select_command(model, input_waters - input_both_waters)
        msg += (
            f'<br>Also, of the waters existing in the input, douse <a href="cxcmd:{sel_dup}">found {ndup}</a>'
            f' and <a href="cxcmd:{sel_xtra}">did not find {nxtra}</a>')

    return model, msg, nnew, compared_waters


# also used in tool.py
def _water_residues(model):
    res = model.residues
    water_res = res[res.names == 'HOH']
    return water_res


def _compare_waters(input_model, output_model, overlap_distance=2):
    '''
    Find how many output waters overlap input waters.
    '''
    # Get water residues in input and output models
    input_waters = _water_residues(input_model)
    output_waters = _water_residues(output_model)

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


def _select_command(model, residues):
    # model to select in is not necessarily the same as the residues
    if len(residues) > 0:
        from chimerax.atomic import concise_residue_spec
        spec = concise_residue_spec(model.session, residues)
        for i, c in enumerate(spec):
            if c not in '#.' and not c.isdigit():
                break
        spec = model.string(style="command") + spec[i:]
    cmd = f'select {spec}' if len(residues) > 0 else 'select clear'
    return cmd, len(residues)


def _add_waters(model, new_wat_res):
    rnum = model.residues.numbers.max(initial = 0) + 1
    res = []
    for r in new_wat_res:
        rc = model.new_residue(r.name, r.chain_id, rnum)
        for a in r.atoms:
            ac = model.new_atom(a.name, a.element)
            ac.coord = a.coord
            ac.draw_mode = ac.STICK_STYLE
            ac.color = (255,0,0,255)
            rc.add_atom(ac)
        res.append(rc)
        rnum += 1
    from chimerax.atomic import Residues
    return Residues(res)


def _show_waters(input_model, douse_model, residue_range, map, map_range):
    m_id = douse_model.id_string
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
    run(douse_model.session, cmd, log=False)
