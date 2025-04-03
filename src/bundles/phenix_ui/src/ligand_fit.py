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

#
# Command to place a ligand in a cryoEM map using Phenix ligandfit.
#
from chimerax.core.tasks import Job
from chimerax.core.errors import UserError
from time import time

class FitJob(Job):

    SESSION_SAVE = False

    def __init__(self, session, executable_location, optional_args, search_center, resolution,
            positional_args, temp_dir, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, search_center, resolution, positional_args,
            temp_dir, verbose, callback, blocking=block)

    def run(self, session, executable_location, optional_args, search_center, resolution, positional_args,
            temp_dir, verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                results = _run_fit_subprocess(session, executable_location, optional_args,
                    search_center, resolution, positional_args, temp_dir, verbose)
            finally:
                self._running = False
            self.session.ui.thread_safe(callback, *results)
        import threading
        thread = threading.Thread(target=threaded_run, daemon=True)
        thread.start()
        super().run()

    def monitor(self):
        from chimerax.core.commands import plural_form
        plural_seconds = lambda n: plural_form(n, "second")
        plural_minutes = lambda n: plural_form(n, "minute")
        delta = int(time() - self.start_t + 0.5)
        if delta < 60:
            time_info = "%d %s" % (delta, plural_seconds(delta))
        elif delta < 3600:
            minutes = delta // 60
            seconds = delta % 60
            time_info = "%d %s and %d %s" % (minutes, plural_minutes(minutes), seconds,
                plural_seconds(seconds))
        else:
            hours = delta // 3600
            minutes = (delta % 3600) // 60
            seconds = delta % 60
            time_info = "%d:%02d:%02d" % (hours, minutes, seconds)
        ses = self.session
        ses.ui.thread_safe(ses.logger.status, "Fitting job still running (%s)" % time_info)

    def next_check(self):
        return self._monitor_interval
        self._monitor_time += self._monitor_interval
        return self._monitor_time

    def running(self):
        return self._running

command_defaults = {
    'verbose': False
}
def phenix_ligand_fit(session, model, ligand, center=None, in_map=None, resolution=None, *, block=None,
        chain_id=None, hbonds=False, phenix_location=None, residue_number=None,
        verbose=command_defaults['verbose'], option_arg=[], position_arg=[]):

    # Find the phenix.ligandfit executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.ligandfit', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

    # some keywords are just to avoid adjacent atom specs, so reassign to more natural names
    search_center = center

    # Setup temporary directory to run phenix.ligandfit
    from tempfile import TemporaryDirectory
    d = TemporaryDirectory(prefix = 'phenix_ligandfit_')  # Will be cleaned up when object deleted.
    temp_dir = d.name

    # Check map_data arg and save map data
    from os import path
    from chimerax.map_data import save_grid_data
    save_grid_data([in_map.data], path.join(temp_dir, 'map.mrc'), session)

    # Save model to file.
    from chimerax.pdb import save_pdb
    save_pdb(session, path.join(temp_dir,'model.pdb'), models=[model], rel_model=in_map)

    if ligand.startswith(('smiles:', 'ccd:', 'pubchem:')):
        ligand_data = ligand
    elif ligand.startswith('file:'):
        ligand_format, ligand_data = ligand.split(':', 1)
    else:
        import os
        if os.path.exists(ligand):
            ligand_data = ligand
            ligand_format = 'file'
        else:
            if ligand.isdigit() and len(ligand) != 3:
                ligand_format = 'pubchem'
            elif len(ligand) in (3,5) and ligand.isalnum():
                ligand_format = 'ccd'
            else:
                ligand_format = 'smiles'
            ligand_data = ligand_format + ':' + ligand
        session.logger.info(f"Guessing ligand format to be '{ligand_format}'")

    try:
        ligand_models, status = session.open_command.open_data(ligand_data)
    except Exception as e:
        raise UserError(f"Cannot open ligand '{ligand}': {str(e)}")

    check_needed = chain_id is not None and residue_number is not None
    if chain_id is None:
        chain_id = sorted(model.residues.unique_chain_ids)[0]
    if residue_number is None:
        residues = model.residues
        residue_number = max(residues.filter(residues.chain_ids == chain_id).numbers) + 1
    if check_needed:
        residues = model.residues
        chain_residues = residues.filter(residues.chain_ids == chain_id)
        match_residues = chain_residues.filter(chain_residues.numbers == residue_number)
        for match_res in match_residues:
            if not match_res.insertion_code:
                raise UserError("A residue with chain_id '%s' and number '%d' already exists in %s" %
                    (chain_id, residue_number, model))

    # Save ligand to file.
    from chimerax.pdb import save_pdb
    save_pdb(session, path.join(temp_dir,'ligand.pdb'), models=ligand_models)

    # Run phenix.ligandfit
    # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
    # the program runs
    callback = lambda placed_ligand, *args, session=session, model=model, chain_id=chain_id, \
        hbonds=hbonds, residue_number=residue_number, d_ref=d: _process_results(
        session, placed_ligand, model, chain_id, residue_number, hbonds)
    FitJob(session, exe_path, option_arg, search_center, resolution, position_arg, temp_dir, verbose,
        callback, block)

class ViewBoxError(ValueError):
    pass

def view_box(session, model):
    """Return the mid-point of the view line of sight intersected with the model bounding box"""
    bbox = model.bounds()
    if bbox is None:
        if not model.display:
            model.display = True
            model.update_drawings()
            bbox = model.bounds()
            model.display = False
        if bbox is None:
            raise ViewBoxError("%s is not displayed" % model)
    min_xyz, max_xyz = bbox.xyz_min, bbox.xyz_max
    #from chimerax.geometry import Plane, ray_segment
    from chimerax.geometry import Plane, PlaneNoIntersectionError
    # X normal, then Y normal, then Z normal planes
    plane_pairs = []
    for fixed_axis, var_axis1, var_axis2 in [(0,1,2), (1,0,2), (2,0,1)]:
        pts1 = [[0,0,0], [0,0,0], [0,0,0], [0,0,0]] # can't do "*4" because you'll get copies
        pts2 = [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]
        for pt1 in pts1:
            pt1[fixed_axis] = min_xyz[fixed_axis]
        for pt2 in pts2:
            pt2[fixed_axis] = max_xyz[fixed_axis]
        for pts in (pts1, pts2):
            pts[0][var_axis1] = min_xyz[var_axis1]
            pts[1][var_axis1] = min_xyz[var_axis1]
            pts[2][var_axis1] = max_xyz[var_axis1]
            pts[3][var_axis1] = max_xyz[var_axis1]

            pts[0][var_axis2] = min_xyz[var_axis2]
            pts[1][var_axis2] = max_xyz[var_axis2]
            pts[2][var_axis2] = min_xyz[var_axis2]
            pts[3][var_axis2] = max_xyz[var_axis2]
        plane_pairs.append((Plane(pts1), Plane(pts2)))
    cam = session.main_view.camera
    origin = cam.position.origin()
    direction = cam.view_direction()
    # find the two points that have the property of being a plane intercept and lying between
    # the other two plane pairs (these two points may not be on opposite sides of the box)
    face_intercepts = []
    for plane_pair in plane_pairs:
        for plane in plane_pair:
            try:
                intersection = plane.line_intersection(origin, direction)
            except PlaneNoIntersectionError:
                continue
            for pp2 in plane_pairs:
                if pp2 is plane_pair:
                    continue
                plane1, plane2 = pp2
                if plane1.distance(intersection) * plane2.distance(intersection) > 0:
                    # outside plane pair
                    break
            else:
                face_intercepts.append(intersection)
    if len(face_intercepts) == 2:
        return (face_intercepts[0] + face_intercepts[1]) / 2
    raise ViewBoxError("Center of view does not intersect %s bounding box" % model)

def _process_results(session, placed_ligand, model, chain_id, residue_number, hbonds):
    session.logger.status("Fitting job finished")
    if model.deleted:
        placed_ligand.delete()
        raise UserError("Receptor structure was deleted during ligand fitting")
    from chimerax.atomic import Atom, colors, Residue
    res = placed_ligand.residues[0]
    res.chain_id = chain_id
    res.number = residue_number
    ligand_atoms = placed_ligand.atoms
    ligand_atoms.draw_modes = Atom.STICK_STYLE
    ligand_atoms.colors = colors.element_colors(ligand_atoms.element_numbers)
    # Assign ligand's secondary structure info, so that when combined
    # it doesn't invalidate the receptor's info
    placed_ligand.residues.ss_types = Residue.SS_COIL
    placed_ligand.ss_assigned = True
    model.combine(placed_ligand, {}, model.scene_position)
    session.logger.info("Ligand added to %s as residue %d in chain %s" % (model,  residue_number, chain_id))
    if hbonds:
        from chimerax.core.commands import run
        run(session, "hbonds %s reveal true" % (model.atomspec + '/' + chain_id + ':' + str(residue_number)))

#NOTE: We don't use a REST server; reference code retained in douse.py

def _run_fit_subprocess(session, exe_path, optional_args, search_center, resolution, positional_args,
        temp_dir, verbose):
    '''
    Run ligandfit in a subprocess and return the ligand.
    '''
    import os
    if hasattr(os, 'sched_getaffinity'):
        processors = max(1, len(os.sched_getaffinity(0)))
    else:
        processors = os.cpu_count()
        if processors is None:
            processors = 1
    if processors == 1:
        procs_arg = []
    else:
        procs_arg = ["nproc=%d" % min(5, processors)]
    from chimerax.core.commands import StringArg
    args = [exe_path] + optional_args + [
            "--json",
            "ligand=ligand.pdb",
            "map_in=map.mrc",
            "model=model.pdb",
            "resolution=%g" % resolution,
            "search_center=%g %g %g" % tuple(search_center.scene_coordinates()),
        ] + procs_arg + positional_args
    tsafe=session.ui.thread_safe
    logger = session.logger
    tsafe(logger.status, f'Running {exe_path} in directory {temp_dir}')
    import subprocess
    p = subprocess.run(args, capture_output = True, cwd = temp_dir)
    if p.returncode != 0:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = (f'phenix.ligandfit exited with error code {p.returncode}\n\n' +
               f'Command: {cmd}\n\n' +
               f'stdout:\n{out}\n\n' +
               f'stderr:\n{err}')
        raise UserError(msg)

    # Log command output
    if verbose:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = f'<pre><b>Command</b>:\n\n{cmd}\n\n<b>stdout</b>:\n\n{out}'
        if err:
            msg += f'\n\n<b>stderr</b>:\n\n{err}'
        msg += '</pre>'
        tsafe(logger.info, msg, is_html=True)

    '''
    from os import path
    json_path = path.join(temp_dir,'LigandFit_run_1_', 'LigandFit_result.json')
    import json
    with open(json_path, 'r') as f:
        print("JSON file contents:\n", f.readlines())
    with open(json_path, 'r') as f:
        info = json.load(f)
    print("ligandfit JSON info:", info)
    '''
    output_marker = "FULL LIGAND MODEL:"
    for line in p.stdout.decode("utf-8").splitlines():
        if line.startswith(output_marker):
            ligand_path = line[len(output_marker):].strip()
            break
    else:
        raise RuntimeError("Could not find ligand file path in ligandFit output")
    return (session.open_command.open_data(ligand_path)[0][0],)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import (CenterArg, OpenFolderNameArg, BoolArg, RepeatOf, StringArg, IntArg,
        OpenFileNameArg, EnumOf, Or, PositiveFloatArg)
    from chimerax.map import MapArg
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        required = [('model', AtomicStructureArg),
                    ('ligand', Or(OpenFileNameArg,StringArg)),
        ],
        required_arguments = ['center', 'in_map', 'resolution'],
        keyword = [('center', CenterArg),
                   ('in_map', MapArg),
                   ('resolution', PositiveFloatArg),
                   # put the above three first so that they show up in usage before the optional keywords
                   ('block', BoolArg),
                   ('chain_id', StringArg),
                   ('hbonds', BoolArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('residue_number', IntArg),
                   ('verbose', BoolArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
        ],
        synopsis = 'Place ligand in map'
    )
    register('phenix ligandFit', desc, phenix_ligand_fit, logger=logger)
