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
    sdev = resolution*sigma_factor
    from chimerax.map.molmap import bounding_grid, add_gaussians
    grid = bounding_grid(xyz, grid_spacing, pad)
    add_gaussians(grid, xyz, weights, sdev, cutoff_range)

    m = grid.full_matrix()

    if level is None:
        # Use level of half the minimum density at point positions
        from chimerax.map_data import interpolate_volume_data
        mxyz, outside = interpolate_volume_data(xyz, grid.xyz_to_ijk_transform, m)
        level = 0.5*min(mxyz)

    from chimerax.map import contour_surface
    va, ta, na = contour_surface(m, level, cap_faces = True, calculate_normals = True)

    # Convert ijk to xyz
    tf = grid.ijk_to_xyz_transform
    tf.transform_points(va, in_place = True)
    tf.transform_normals(na, in_place = True)
    from chimerax.geometry import vector
    vector.normalize_vectors(na)

    return va, na, ta, level
