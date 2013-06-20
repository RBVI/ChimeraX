# -----------------------------------------------------------------------------
# Optimize placement of 2 or more models, fitting one at a time with others
# subtracted from map.
#
def fit_sequence(models, volume, steps,
                 envelope = True, metric = 'overlap',
                 optimize_translation = True, optimize_rotation = True,
                 max_steps = 2000,
                 ijk_step_size_min = 0.01, ijk_step_size_max = 0.5,
                 request_stop_cb = None):

    from chimera import Xform
    data_array, xyz_to_ijk_transform = \
        volume.matrix_and_transform(Xform(), subregion = None, step = 1)

    size = tuple(data_array.shape[::-1])
    from numpy import float32, multiply, add, array
    from VolumeData import grid_indices
    grid_points = grid_indices(size, float32)
    import Matrix as M
    M.transform_points(grid_points, M.invert_matrix(xyz_to_ijk_transform))
    grid_points_to_world_xform = Xform()

    # Make float32 copy for subtracting interpolated molecule maps.
    d = array(data_array, float32)

    from move import position_history as h
    h.record_position(models, [], volume)

    fits = {}
    import fitmap as F
    from VolumeViewer.volume import minimum_rms_scale
    from search import Fit
    for s in range(steps):
        mi = s % len(models)
        v = models[mi]
        if request_stop_cb('Fitting %s in %s, step %d'
                           % (v.name, volume.name, s+1)):
            return
        # Subtract off other maps.
        d[:] = data_array
        for m in models:
            if not m is v:
                values = m.interpolated_values(grid_points, grid_points_to_world_xform, subregion = None, step = None)
                level = min(m.surface_levels)
                scale = minimum_rms_scale(values, data_array.ravel(), level)
                multiply(values, scale, values)
                add(d, values.reshape(d.shape), d)
        # Fit in difference map.
        points, point_weights = F.map_points_and_weights(v, envelope)
        move_tf, stats = F.locate_maximum(
            points, point_weights, d, xyz_to_ijk_transform,
            max_steps, ijk_step_size_min, ijk_step_size_max,
            optimize_translation, optimize_rotation, metric,
            request_stop_cb = None)
        v.openState.globalXform(M.chimera_xform(move_tf))
        fits[v] = Fit([v], None, volume, stats)

    h.record_position(models, [], volume)

    flist = [fits[m] for m in models if m in fits]
    return flist

