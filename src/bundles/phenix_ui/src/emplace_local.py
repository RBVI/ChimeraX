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
# Command to place a structure in a cryoEM map using Phenix emplace_local.
#
from chimerax.core.tasks import Job
from chimerax.core.errors import UserError
from time import time

class FitJob(Job):

    SESSION_SAVE = False

    def __init__(self, session, executable_location, optional_args, map1_file_name,
            map2_file_name, search_center, model_file_name, prefitted_file_name,
            positional_args, temp_dir, resolution, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        self.start(session, executable_location, optional_args, map1_file_name,
            map2_file_name, search_center, model_file_name, prefitted_file_name, positional_args,
            temp_dir, resolution, verbose, callback, blocking=block)

    def run(self, session, executable_location, optional_args, map1_file_name,
            map2_file_name, search_center, model_file_name, prefitted_file_name, positional_args,
            temp_dir, resolution, verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                results = _run_fit_subprocess(session, executable_location, optional_args,
                    map1_file_name, map2_file_name, search_center, model_file_name, prefitted_file_name,
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
def phenix_local_fit(session, model, center=None, map_data=None, *, resolution=0.0, show_sharpened_map=False,
        apply_symmetry=False, prefitted=None, show_tool=True, block=None, phenix_location=None,
        verbose=command_defaults['verbose'], option_arg=[], position_arg=[]):

    # Find the phenix.voyager.emplace_local executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.voyager.emplace_local', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

    # some keywords are just to avoid adjacent atom specs, so reassign to more natural names
    search_center = center

    # Setup temporary directory to run phenix.voyager.emplace_local.
    from tempfile import TemporaryDirectory
    d = TemporaryDirectory(prefix = 'phenix_emis_')  # Will be cleaned up when object deleted.
    temp_dir = d.name

    # Check map_data arg and save map data
    from os import path
    from chimerax.map_data import save_grid_data
    if len(map_data) == 1:
        if resolution == 0.0:
            raise UserError("When fitting into a full map, the map resolution must be specified")
        map_arg1, map_arg2 = 'full_map.mrc', None
        save_grid_data([map_data[0].data], path.join(temp_dir, map_arg1), session)
    elif len(map_data) == 2:
        map_arg1, map_arg2 = 'half_map1.mrc', 'half_map2.mrc'
        save_grid_data([map_data[0].data], path.join(temp_dir, map_arg1), session)
        save_grid_data([map_data[1].data], path.join(temp_dir, map_arg2), session)
    else:
        raise UserError("Please specify two half maps or one full map.  You specified %d maps"
            % (len(map_data)))

    # Emplace_local handles non-zero origins, so don't have to write an adjusted map

    # Save model to file.
    from chimerax.pdb import save_pdb
    save_pdb(session, path.join(temp_dir,'model.pdb'), models=[model], rel_model=map_data[0])

    # Save prefitted models to combined file
    if prefitted is None:
        prefitted_arg = None
    else:
        from chimerax.atomic.cmd import combine_cmd
        combo = combine_cmd(session, prefitted, add_to_session=False)
        prefitted_arg = 'prefitted.pdb'
        save_pdb(session, path.join(temp_dir, prefitted_arg), models=[combo], rel_model=map_data[0])

    # Run phenix.voyager.emplace_local
    # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
    # the program runs
    callback = lambda transforms, sharpened_maps, llgs, ccs, *args, session=session, maps=map_data, \
        ssm=show_sharpened_map, app_sym=apply_symmetry, show_tool=show_tool, d_ref=d: _process_results(
        session, transforms, sharpened_maps, llgs, ccs, model, maps, ssm, app_sym, show_tool)
    FitJob(session, exe_path, option_arg, map_arg1, map_arg2, search_center,
        "model.pdb", prefitted_arg, position_arg, temp_dir, resolution, verbose, callback, block)

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

def _process_results(session, transforms, map_paths, llgs, ccs, orig_model, maps, show_sharpened_map,
        apply_symmetry, show_tool):
    session.logger.status("Fitting job finished")
    if orig_model.deleted:
        raise UserError("Structure being fitting was deleted during fitting")
    sharpened_maps = []
    for map_path in map_paths:
        sharpened_map, status = session.open_command.open_data(map_path)
        sharpened_maps.extend(sharpened_map)
    if len(transforms) > 1:
        for i, sharpened_map in enumerate(sharpened_maps):
            sharpened_map.name = "map %d" % (i+1)
            sharpened_map.display = show_sharpened_map and i == 0
        from chimerax.core.models import Model
        group = Model("sharpened local maps", session)
        group.add(sharpened_maps)
        session.models.add([group])
    else:
        sharpened_map = sharpened_maps[0]
        sharpened_map.name = "sharpened local map"
        sharpened_map.display = show_sharpened_map
        session.models.add(sharpened_maps)
    from chimerax.core.commands import run, concise_model_spec, StringArg
    if apply_symmetry:
        sym_map = maps[0]
        if sym_map.deleted:
            session.logger.warning("Map being fitted has been deleted; not applying symmetry")
            apply_symmetry = False
        else:
            run(session, "measure symmetry " + sym_map.atomspec)
            if not sym_map.data.symmetries:
                session.logger.warning(
                    'Could not determine symmetry for %s<br><br>'
                    'If you know the symmetry of the map, you can create symmetry copies of the structure'
                    ' with the <a href="help:user/commands/sym.html">sym</a> command and then combine the'
                    ' symmetry copies into a single structure with the <a'
                    ' href="help:user/commands/combine.html">combine</a> command'
                    % sym_map, is_html=True)
                apply_symmetry = False
    if show_tool and len(transforms) > 1 and session.ui.is_gui:
        from .tool import EmplaceLocalResultsViewer
        EmplaceLocalResultsViewer(session, orig_model, transforms, llgs, ccs, show_sharpened_map, group,
            sym_map if apply_symmetry else None)
    else:
        from chimerax.geometry import Place
        orig_model.scene_position = Place(transforms[0]) * orig_model.scene_position
        if apply_symmetry:
            modelspec = orig_model.atomspec
            prev_models = set(session.models[:])
            run(session, f"sym {modelspec} symmetry {sym_map.atomspec} copies true")
            added = [m for m in session.models if m not in prev_models]
            orig_id, orig_name = orig_model.id[0], orig_model.name
            run(session, f"close {modelspec}")
            run(session, "combine " + concise_model_spec(session, added) + " close true"
                " modelId %d name %s" % (orig_id, StringArg.unparse(orig_name)))

#NOTE: We don't use a REST server; reference code retained in douse.py

def _run_fit_subprocess(session, exe_path, optional_args, map1_file_name, map2_file_name, search_center,
        model_file_name, prefitted_file_name, positional_args, temp_dir, resolution, verbose):
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
    if prefitted_file_name is None:
        prefitted_arg = []
    else:
        prefitted_arg = [ "fixed_model_file=%s" % prefitted_file_name ]
    args = [exe_path] + optional_args + map_args + [
            "d_min=%g" % resolution,
            "model_file=%s" % StringArg.unparse(model_file_name),
            "sphere_center=(%g,%g,%g)" % tuple(search_center.scene_coordinates()),
            "--json",
        ] + prefitted_arg + positional_args
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
    from chimerax.core.commands import plural_form
    num_solutions = info["n_solutions"]
    tsafe(logger.info, "%d fitting %s" % (num_solutions, plural_form(num_solutions, "solution")))
    tsafe(logger.info, "map LLG %s: %s" % (plural_form(num_solutions, "value"),
        ', '.join(["%g" % v for v in info["mapLLG"]])))
    tsafe(logger.info, "map CC %s: %s" % (plural_form(num_solutions, "value"),
        ', '.join(["%g" % v for v in info["mapCC"]])))
    if 'map_filenames' in info:
        # Phenix 2.0
        map_paths = [path.join(temp_dir, mf) for mf in info["map_filenames"]]
    else:
        # Phenix 1.2
        map_paths = [path.join(temp_dir, info["map_filename"])]
        info['RT'] = info['RT'][:1]
        info['mapLLG'] = info['mapLLG'][:1]
        info['mapCC'] = info['mapCC'][:1]

    from numpy import array
    return [array(rt) for rt in info['RT']], map_paths, info["mapLLG"], info["mapCC"]

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import (CenterArg, OpenFolderNameArg, BoolArg, NonNegativeFloatArg,
        RepeatOf, StringArg)
    from chimerax.map import MapArg, MapsArg
    from chimerax.atomic import AtomicStructureArg, AtomicStructuresArg
    desc = CmdDesc(
        required = [('model', AtomicStructureArg),
        ],
        required_arguments = ['center', 'map_data'],
        keyword = [('block', BoolArg),
                   ('center', CenterArg),
                   ('map_data', MapsArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('verbose', BoolArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
                   ('prefitted', AtomicStructuresArg),
                   ('resolution', NonNegativeFloatArg),
                   ('show_sharpened_map', BoolArg),
                   ('apply_symmetry', BoolArg),
                   ('show_tool', BoolArg),
        ],
        synopsis = 'Place structure in map'
    )
    register('phenix emplaceLocal', desc, phenix_local_fit, logger=logger)
