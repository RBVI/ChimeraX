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

# -----------------------------------------------------------------------------
# Optimize placement of 2 or more models, fitting one at a time with others
# subtracted from map.
#
def fit_sequence(models, volume, steps, subtract_maps = [],
                 envelope = True, include_zeros = False, metric = 'overlap',
                 optimize_translation = True, optimize_rotation = True,
                 max_steps = 2000,
                 ijk_step_size_min = 0.01, ijk_step_size_max = 0.5,
                 request_stop_cb = None, log = None):

    from chimerax.geometry import identity
    data_array, xyz_to_ijk_transform = \
        volume.matrix_and_transform(identity(), subregion = None, step = 1)

    size = tuple(data_array.shape[::-1])
    from numpy import float32, multiply, add, array
    from chimerax.map_data import grid_indices
    grid_points = grid_indices(size, float32)
    xyz_to_ijk_transform.inverse().transform_points(grid_points, in_place = True)
    grid_points_to_scene_transform = identity()

    # Make float32 copy for subtracting interpolated molecule maps.
    d = array(data_array, float32)

    from chimerax.atomic import Atoms
    from .move import position_history as h
    h.record_position(models, Atoms(), volume)

    fits = {}
    from . import fitmap as F
    from chimerax.map.volume import minimum_rms_scale
    from .search import Fit
    for s in range(steps):
        mi = s % len(models)
        v = models[mi]
        if request_stop_cb and request_stop_cb('Fitting %s in %s, step %d'
                                               % (v.name, volume.name, s+1)):
            return
        # Subtract off other maps.
        d[:] = data_array
        for m in models + subtract_maps:
            if not m is v:
                values = m.interpolated_values(grid_points, grid_points_to_scene_transform,
                                               subregion = None, step = None)
                level = m.minimum_surface_level
                scale = minimum_rms_scale(values, data_array.ravel(), level)
                multiply(values, scale, values)
                add(d, values.reshape(d.shape), d)
        # Fit in difference map.
        points, point_weights = F.map_points_and_weights(v, envelope, include_zeros = include_zeros)
        move_tf, stats = F.locate_maximum(
            points, point_weights, d, xyz_to_ijk_transform,
            max_steps, ijk_step_size_min, ijk_step_size_max,
            optimize_translation, optimize_rotation, metric,
            request_stop_cb = None)
        v.position = move_tf * v.position
        fits[v] = Fit([v], None, volume, stats)
        if log:
            log.info(F.map_fit_message(v, volume, stats))


    h.record_position(models, Atoms(), volume)

    flist = [fits[m] for m in models if m in fits]
    return flist

