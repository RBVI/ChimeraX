def gaussian_surface(xyz, weights, resolution, level = None, grid_spacing = None):
    '''
    Return vertex, normal vector and triangles for a contour surface computed
    from a density map created as a sum of Gaussians at specified center positions
    with specified heights, and Gaussian width determined by resolution.
    If the contour level is None then compute the lowest density value at
    the center point positions.  Return the input level or this computed level
    as a fourth return value.
    '''
    if grid_spacing is None:
        grid_spacing = 0.25 * resolution

    from math import pow, pi
    pad = 3*resolution
    cutoff_range = 5
    from math import pi, sqrt, pow
    sigma_factor = 1 / (pi*sqrt(2))
    from ..map import molmap
    grid = molmap.gaussian_grid_data(xyz, weights, resolution, grid_spacing,
                                     pad, cutoff_range, sigma_factor)

    m = grid.full_matrix()

    if level is None:
        # Use level of half the minimum density at point positions
        from ..map import data
        mxyz, outside = data.interpolate_volume_data(xyz, grid.xyz_to_ijk_transform, m)
        level = 0.5*min(mxyz)

    from ..map import contour_surface
    va, ta, na = contour_surface(m, level, cap_faces = True, calculate_normals = True)

    # Convert ijk to xyz
    tf = grid.ijk_to_xyz_transform
    tf.move(va)
    tf.zero_translation().move(na)
    from ..geometry import vector
    vector.normalize_vectors(na)

    return va, na, ta, level
