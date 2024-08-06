# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

    def __init__(self, session, executable_location, optional_args, map1_file_name,
            map2_file_name, search_center, model_file_name,
            positional_args, temp_dir, resolution, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, map1_file_name,
            map2_file_name, search_center, model_file_name, positional_args, temp_dir, resolution,
            verbose, callback, blocking=block)

    def run(self, session, executable_location, optional_args, map1_file_name,
            map2_file_name, search_center, model_file_name, positional_args, temp_dir, resolution,
            verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                results = _run_fit_subprocess(session, executable_location, optional_args,
                    map1_file_name, map2_file_name, search_center, model_file_name,
                    positional_args, temp_dir, resolution, verbose)
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
def phenix_ligand_fit(session, model, ligand, center=None, in_map=None, *, block=None,
        phenix_location=None, verbose=command_defaults['verbose'],
        option_arg=[], position_arg=[]):

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
    elif ligand,startswith('file:')
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
        ligand_model = session.open_command.open_data(ligand_data)
    except Exception as e:
        raise UserError(f"Cannot open ligand '{ligand}': {str(e)}")

    # Save ligand to file.
    from chimerax.pdb import save_pdb
    save_pdb(session, path.join(temp_dir,'ligand.pdb'), models=[ligand_model])

    # Run phenix.ligandfit
    # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
    # the program runs
    callback = lambda transform, sharpened_map, *args, session=session, maps=map_data, \
        ssm=show_sharpened_map, app_sym=apply_symmetry, d_ref=d: _process_results(session,
        transform, sharpened_map, model, maps, ssm, app_sym)
    FitJob(session, exe_path, option_arg, map_arg1, map_arg2, search_center,
        "model.pdb", position_arg, temp_dir, resolution, verbose, callback, block)

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

def _process_results(session, transform, sharpened_map, orig_model, maps, show_sharpened_map,
        apply_symmetry):
    if orig_model.deleted:
        raise UserError("Structure being fitting was deleted during fitting")
    from chimerax.geometry import Place
    orig_model.scene_position = Place(transform) * orig_model.scene_position
    sharpened_map.name = "sharpened local map"
    sharpened_map.display = show_sharpened_map
    session.models.add([sharpened_map])
    session.logger.status("Fitting job finished")
    if apply_symmetry:
        sym_map = maps[0]
        if sym_map.deleted:
            raise UserError("Map being fitted has been deleted; not applying symmetry")
        from chimerax.core.commands import run, concise_model_spec, StringArg
        run(session, "measure symmetry " + sym_map.atomspec)
        if maps[0].data.symmetries:
            prev_models = set(session.models[:])
            run(session, "sym " + orig_model.atomspec + " symmetry " + sym_map.atomspec + " copies true")
            added = [m for m in session.models if m not in prev_models]
            run(session, "combine " + concise_model_spec(session, [orig_model] + added) + " close true"
                " modelId %d name %s" % (orig_model.id[0], StringArg.unparse(orig_model.name)))
        else:
            session.logger.warning(
                'Could not determine symmetry for %s<br><br>'
                'If you know the symmetry of the map, you can create symmetry copies of the structure'
                ' with the <a href="help:user/commands/sym.html">sym</a> command and then combine the'
                ' symmetry copies with the original structure with the <a'
                ' href="help:user/commands/combine.html">combine</a> command'
                % sym_map, is_html=True)

#NOTE: We don't use a REST server; reference code retained in douse.py

def _run_fit_subprocess(session, exe_path, optional_args, map1_file_name,
        map2_file_name, search_center, model_file_name, positional_args, temp_dir, resolution, verbose):
    '''
    Run emplace_local in a subprocess and return the model.
    '''
    from chimerax.core.commands import StringArg
    if map2_file_name is None:
        map_args = ["map=%s" % StringArg.unparse(map1_file_name)]
    else:
        map_args = [
            "map1=%s" % StringArg.unparse(map1_file_name),
            "map2=%s" % StringArg.unparse(map2_file_name),
        ]
    args = [exe_path] + optional_args + map_args + [
            "d_min=%g" % resolution,
            "model_file=%s" % StringArg.unparse(model_file_name),
            "sphere_center=(%g,%g,%g)" % tuple(search_center.scene_coordinates()),
            "--json",
        ] + positional_args
    tsafe=session.ui.thread_safe
    logger = session.logger
    tsafe(logger.status, f'Running {exe_path} in directory {temp_dir}')
    import subprocess
    p = subprocess.run(args, capture_output = True, cwd = temp_dir)
    if p.returncode != 0:
        cmd = " ".join(args)
        out, err = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        msg = (f'phenix.voyager.emplace_local exited with error code {p.returncode}\n\n' +
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

    # Open new model with added waters
    from os import path
    json_path = path.join(temp_dir,'emplace_local_result.json')
    import json
    with open(json_path, 'r') as f:
        info = json.load(f)
    model_path = path.join(temp_dir, info["model_filename"])
    map_path = path.join(temp_dir, info["map_filename"])
    sharpened_maps, status = session.open_command.open_data(map_path)
    from chimerax.core.commands import plural_form
    num_solutions = info["n_solutions"]
    tsafe(logger.info, "%d fitting %s" % (num_solutions, plural_form(num_solutions, "solution")))
    tsafe(logger.info, "map LLG %s: %s" % (plural_form(num_solutions, "value"),
        ', '.join(["%g" % v for v in info["mapLLG"]])))
    tsafe(logger.info, "map CC %s: %s" % (plural_form(num_solutions, "value"),
        ', '.join(["%g" % v for v in info["mapCC"]])))

    from numpy import array
    return array(info['RT'][0]), sharpened_maps[0]

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import (CenterArg, OpenFolderNameArg, BoolArg, RepeatOf, StringArg,
        OpenFileNameArg, EnumOf)
    from chimerax.map import MapArg
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        required = [('model', AtomicStructureArg),
                    ('ligand', Or(OpenFileNameArg,StringArg)),
        ],
        required_arguments = ['center', 'in_map'],
        keyword = [('block', BoolArg),
                   ('center', CenterArg),
                   ('in_map', MapArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('verbose', BoolArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
        ],
        synopsis = 'Place ligand in map'
    )
    register('phenix ligandFit', desc, phenix_ligand_fit, logger=logger)
