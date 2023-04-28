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
# Command to fit loops guided by cryoEM maps using Phenix fit_loops.
#
from chimerax.core.tasks import Job
from time import time

class FitLoopsJob(Job):

    SESSION_SAVE = False

    def __init__(self, session, executable_location, optional_args, map_file_name, model_file_name,
            positional_args, temp_dir, start_res_number, chain_id, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, map_file_name, model_file_name,
            positional_args, temp_dir, start_res_number, chain_id, verbose, callback, blocking=block)

    def run(self, session, executable_location, optional_args, map_file_name, model_file_name,
            positional_args, temp_dir, start_res_number, chain_id, verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                results = _run_fit_loops_subprocess(session, executable_location, optional_args,
                    map_file_name, model_file_name, positional_args, temp_dir, start_res_number,
                    chain_id, verbose)
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
        ses.ui.thread_safe(ses.logger.status, "Fit loops job still running (%s)" % time_info)

    def next_check(self):
        return self._monitor_interval
        self._monitor_time += self._monitor_interval
        return self._monitor_time

    def running(self):
        return self._running

command_defaults = {
    'processors': None,
    'replace': True,
    'verbose': False
}
def phenix_fit_loops(session, structure, map, *, block=None, phenix_location=None,
        chain_id=None,
        processors=command_defaults['processors'],
        replace=command_defaults['replace'],
        start_res_number=None,
        sequence_file=None,
        verbose=command_defaults['verbose'],
        option_arg=[], position_arg=[]):

    # Find the phenix.fit_loops executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.fit_loops', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

    # Setup temporary directory to run phenix.fit_loops.
    from tempfile import TemporaryDirectory
    d = TemporaryDirectory(prefix = 'phenix_fit_loops_')  # Will be cleaned up when object deleted.
    temp_dir = d.name

    # Save map to file
    from os import path
    from chimerax.map_data import save_grid_data
    save_grid_data([map.data], path.join(temp_dir,'map.mrc'), session)

    # Guessing that like douse, fit_loops ignores the MRC file origin so if it is non-zero
    # shift the atom coordinates so they align with the origin 0 map.
    map_0, shift = _fix_map_origin(map)

    # Save structure to file.
    from chimerax.pdb import save_pdb
    save_pdb(session, path.join(temp_dir,'model.pdb'), models=[structure], rel_model=map_0)

    # Run phenix.fit_loops
    # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
    # fit_loops runs
    callback = lambda fit_loops_model, *args, session=session, shift=shift, structure=structure, \
        map=map, start_res_number=start_res_number, replace=replace, chain_id=chain_id, d_ref=d:\
        _process_results(session, fit_loops_model, map, shift, structure, start_res_number, replace,
        chain_id)
    FitLoopsJob(session, exe_path, option_arg, "map.mrc", "model.pdb", position_arg, temp_dir,
        start_res_number, chain_id, verbose, callback, block)

def _process_results(session, fit_loops_model, map, shift, structure, start_res_number, replace, chain_id):
    fit_loops_model.position = map.scene_position
    if shift is not None:
        fit_loops_model.atoms.coords += shift

    #TODO
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
            DouseResultsViewer(session, "Water Placement Results", near_model, model, compared_waters, map)

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


def _run_fit_loops_subprocess(session, exe_path, optional_args, map_filename, model_filename,
        positional_args, temp_dir, start_res_number, chain_id, verbose):
    '''
    Run fit_loops in a subprocess and return the model with predicted waters.
    '''
    args = [exe_path] + optional_args + [f"map_in={map_filename}", f"pdb_in={model_filename}"] \
        + positional_args
    if start_res_number is not None:
        args += [f"start={start_res_number}"]
    if chain_id is not None:
        args += [f"chain_id={chain_id}"]
    tsafe=session.ui.thread_safe
    logger = session.logger
    tsafe(logger.status, f'Running {exe_path} in directory {temp_dir}')
    import subprocess
    p = subprocess.run(args, capture_output = True, cwd = temp_dir)
    if p.returncode != 0:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = (f'phenix.fit_loops exited with error code {p.returncode}\n\n' +
               f'Command: {cmd}\n\n' +
               f'stdout:\n{out}\n\n' +
               f'stderr:\n{err}')
        from chimerax.core.errors import UserError
        raise UserError(msg)

    # Log phenix fit_loops command output
    if verbose:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = f'<pre><b>Command</b>:\n\n{cmd}\n\n<b>stdout</b>:\n\n{out}'
        if err:
            msg += f'\n\n<b>stderr</b>:\n\n{err}'
        msg += '</pre>'
        tsafe(logger.info, msg, is_html=True)

    # Open new model with added loops
    from os import path
    output_path = path.join(temp_dir,'connect.pdb')
    from chimerax.pdb import open_pdb
    models, info = open_pdb(session, output_path, log_info = False)

    return models[0]

"""
def _copy_new_waters(douse_model, near_model, keep_input_water, far_water, map_name):
    model = near_model.copy()
    model.position = douse_model.position
    model.name = near_model.name + ' douse'

    # Add found water molecules to copy of input molecule.
    if keep_input_water:
        from chimerax.check_waters import compare_waters
        input_waters, douse_only_waters, douse_both_waters, input_both_waters = compare_waters(near_model,
            douse_model)
        # since the douse model will be closed, translate its "both" waters to the new model's waters
        res_map =  { r:i for i, r in enumerate(near_model.residues) }
        from chimerax.atomic import Residues
        both_waters = Residues([model.residues[res_map[r]] for r in input_both_waters])
    else:
        douse_only_waters = _water_residues(douse_model)
        _water_residues(model).delete()
        compared_waters = None
    added_wat_res = _add_waters(model, douse_only_waters)
    if keep_input_water:
        compared_waters = (input_waters, added_wat_res, both_waters, input_both_waters)

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
"""

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import OpenFolderNameArg, OpenFileNameArg, BoolArg, FloatArg, RepeatOf
    from chimerax.core.commands import StringArg, IntArg, TupleOF
    from chimerax.map import MapArg
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        required = [('structure', AtomicStructureArg), ('map', MapArg)],
        keyword = [('block', BoolArg),
                   ('chain_id', StringArg),
                   ('processors', IntArg),
                   ('replace', BoolArg),
                   ('sequence_file', OpenFileNameArg),
                   ('start_res_number', IntArg),
                   ('verbose', BoolArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
        ],
        synopsis = 'Fit loop(s) into density'
    )
    register('phenix fitLoops', desc, phenix_fit_loops, logger=logger)
