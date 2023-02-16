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
# Command to place a structure in a cryoEM map using Phenix emplace_local.
#
from chimerax.core.tasks import Job
from chimerax.core.errors import UserError
from time import time

class FitJob(Job):

    SESSION_SAVE = False

    #def __init__(self, session, executable_location, optional_args, map_file_name, half_map1_file_name,
    def __init__(self, session, executable_location, optional_args, half_map1_file_name,
            half_map2_file_name, search_center, model_file_name,
            positional_args, temp_dir, resolution, verbose, callback, block):
        super().__init__(session)
        self._running = False
        self._monitor_time = 0
        self._monitor_interval = 10
        #self.start(session, executable_location, optional_args, map_file_name, half_map1_file_name,
        self.start(session, executable_location, optional_args, half_map1_file_name,
            half_map2_file_name, search_center, model_file_name, positional_args, temp_dir, resolution,
            verbose, callback, blocking=block)

    #def run(self, session, executable_location, optional_args, map_file_name, half_map1_file_name,
    def run(self, session, executable_location, optional_args, half_map1_file_name,
            half_map2_file_name, search_center, model_file_name, positional_args, temp_dir, resolution,
            verbose, callback, **kw):
        self._running = True
        self.start_t = time()
        def threaded_run(self=self):
            try:
                #results = _run_fit_subprocess(session, executable_location, optional_args, map_file_name,
                results = _run_fit_subprocess(session, executable_location, optional_args,
                    half_map1_file_name, half_map2_file_name, search_center, model_file_name,
                    positional_args, temp_dir, resolution, verbose)
            finally:
                self._running = False
            self.session.ui.thread_safe(callback, *results)
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
def phenix_local_fit(session, model, center=None, half_maps=None, resolution=None, *,
        block=None, phenix_location=None, verbose=command_defaults['verbose'],
        option_arg=[], position_arg=[]):

    # Find the phenix.voyager.emplace_local executable
    from .locate import find_phenix_command
    exe_path = find_phenix_command(session, 'phenix.voyager.emplace_local', phenix_location)

    # if blocking not explicitly specified, block if in a script or in nogui mode
    if block is None:
        block = session.in_script or not session.ui.is_gui

    # some keywords are just to avoid adjacent atom specs, so reassign to more natural names
    search_center = center

    if len(half_maps) != 2:
        raise UserError("Please specify exactly two half maps.  You specified %d" % (len(half_maps)))

    # Setup temporary directory to run phenix.voyager.emplace_local.
    from tempfile import TemporaryDirectory
    d = TemporaryDirectory(prefix = 'phenix_emis_')  # Will be cleaned up when object deleted.
    temp_dir = d.name

    # Save maps to files
    from os import path
    from chimerax.map_data import save_grid_data
    #save_grid_data([target_map.data], path.join(temp_dir,'target_map.mrc'), session)
    save_grid_data([half_maps[0].data], path.join(temp_dir,'half_map1.mrc'), session)
    save_grid_data([half_maps[1].data], path.join(temp_dir,'half_map2.mrc'), session)

    # Emplace_local ignores the MRC file origin so if it is non-zero
    # shift the atom coordinates so they align with the origin 0 map.
    #map_0, shift = _fix_map_origin(target_map)
    map_0, shift = _fix_map_origin(half_maps[0])

    # Save model to file.
    from chimerax.pdb import save_pdb
    save_pdb(session, path.join(temp_dir,'model.pdb'), models=[model], rel_model=map_0)

    # Run phenix.voyager.emplace_local
    # keep a reference to 'd' in the callback so that the temporary directory isn't removed before
    # the program runs
    callback = lambda fit_model, sharpened_map, *args, session=session, half_maps=half_maps, shift=shift, d_ref=d: \
        _process_results(session, fit_model, sharpened_map, model, half_maps, shift)
    #FitJob(session, exe_path, option_arg, "target_map.mrc", "half_map1.mrc", "half_map2.mrc", search_center,
    #    "model.pdb", position_arg, temp_dir, resolution, verbose, callback, block)
    FitJob(session, exe_path, option_arg, "half_map1.mrc", "half_map2.mrc", search_center,
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

def _process_results(session, fit_model, sharpened_map, orig_model, half_maps, shift):
    fit_model.position = half_maps[0].scene_position
    if shift is not None:
        fit_model.atoms.coords += shift
    from chimerax.std_commands.align import align
    xf = align(session, orig_model.atoms, fit_model.atoms, log_info=False)[-1]
    fit_model.delete()
    sharpened_map.name = "sharpened local map"
    session.models.add([sharpened_map])
    session.logger.status("Fitting job finished")
    #session.logger.info("Fitted model opened as %s" % fit_model)

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

#NOTE: We don't use a REST server; reference code retained in douse.py

#def _run_fit_subprocess(session, exe_path, optional_args, map_file_name, half_map1_file_name,
def _run_fit_subprocess(session, exe_path, optional_args, half_map1_file_name,
        half_map2_file_name, search_center, model_file_name, positional_args, temp_dir, resolution, verbose):
    '''
    Run emplace_local in a subprocess and return the model.
    '''
    from chimerax.core.commands import StringArg
    args = [exe_path] + optional_args + [
            #"map=%s" % StringArg.unparse(map_file_name),
            "map1=%s" % StringArg.unparse(half_map1_file_name),
            "map2=%s" % StringArg.unparse(half_map2_file_name),
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
    from chimerax.pdb import open_pdb
    fitted_models, status = open_pdb(session, model_path, log_info=False)
    map_path = path.join(temp_dir, info["map_filename"])
    sharpened_maps, status = session.open_command.open_data(map_path)
    from chimerax.core.commands import plural_form
    num_solutions = info["n_solutions"]
    tsafe(logger.info, "%d fitting %s" % (num_solutions, plural_form(num_solutions, "solution")))
    tsafe(logger.info, "map LLG %s: %s" % (plural_form(num_solutions, "value"),
        ', '.join(["%g" % v for v in info["mapLLG"]])))
    tsafe(logger.info, "map CC %s: %s" % (plural_form(num_solutions, "value"),
        ', '.join(["%g" % v for v in info["mapCC"]])))

    return fitted_models[0], sharpened_maps[0]

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import CenterArg, OpenFolderNameArg, BoolArg, FloatArg, RepeatOf, StringArg
    from chimerax.map import MapArg, MapsArg
    from chimerax.atomic import AtomicStructureArg, AtomicStructuresArg
    desc = CmdDesc(
        required = [('model', AtomicStructureArg),
        ],
        #required_arguments = ['center', 'half_maps', 'in_map', 'resolution'],
        required_arguments = ['center', 'half_maps', 'resolution'],
        keyword = [('block', BoolArg),
                   ('center', CenterArg),
                   ('half_maps', MapsArg),
                   ('phenix_location', OpenFolderNameArg),
                   ('verbose', BoolArg),
                   ('option_arg', RepeatOf(StringArg)),
                   ('position_arg', RepeatOf(StringArg)),
                   ('resolution', FloatArg),
        ],
        synopsis = 'Place structure in map'
    )
    register('phenix emplaceLocal', desc, phenix_local_fit, logger=logger)
