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
# Make a new density map by unrolling a cylinder with inner radius r0, outer
# radius r1, and height h by cutting a seam parallel to the axis and flattening
# the cylindrical sheet.
#
def unroll_operation(v, r0, r1, h, center, axis, gsp, subregion, step, modelId):

    from math import ceil, pi
    zsize = int(max(1,ceil(h/gsp)))                     # cylinder height
    xsize = int(max(1,ceil((r1-r0)/gsp)))               # slab thickness
    rmid = 0.5 * (r0 + r1)
    circum = rmid * 2 * pi
    ysize = int(max(1,ceil(circum / gsp)))              # circumference

    from chimerax.geometry import normalize_vector
    axis = normalize_vector(axis)
    agrid_points = annulus_grid(r0, r1, center, axis, ysize, xsize)
    grid_points = agrid_points.reshape((ysize*xsize,3))
    grid_points[:] += tuple([-0.5*h*ai for ai in axis]) # Shift annulus.
    from numpy import empty
    values = empty((zsize, ysize, xsize), v.data.value_type)
    axis_step = tuple([h*float(ai)/(zsize-1) for ai in axis])
    for i in range(zsize):
        vval = v.interpolated_values(grid_points,
                                     subregion = subregion, step = step)
        values[i,:,:] = vval.reshape((ysize,xsize))
        grid_points[:] += axis_step                     # Shift annulus.

    from chimerax.map_data import ArrayGridData
    gstep = (float(r1-r0)/(xsize-1), circum/(ysize-1), float(h)/(zsize-1))
    gorigin = (center[0]+r0, center[1]-0.5*circum, center[2]-0.5*h)
    g = ArrayGridData(values, gorigin, gstep, name = 'unrolled %s' % v.name)
    from chimerax.map import volume_from_grid_data
    vu = volume_from_grid_data(g, v.session, model_id = modelId)
    vu.copy_settings_from(v, copy_region = False, copy_active = False, copy_colors = False)

    if axis[0] != 0 or axis[1] != 0:
        # Rotate so unrolled volume is tangential to cylinder
        from chimerax.geometry import orthonormal_frame
        vu.position = v.position * orthonormal_frame(axis)

    return vu
    
# -----------------------------------------------------------------------------
#
def annulus_grid(radius0, radius1, center, axis, ncircum, nradius):

    from math import pi, cos, sin
    from numpy import empty, float32, multiply
    grid_points = empty((ncircum, nradius, 3), float32)
    for i in range(ncircum):
        a = -pi + 2*pi*float(i)/ncircum
        grid_points[i,0,:] = (cos(a), sin(a), 0)
    for i in range(nradius):
        grid_points[:,i,:] = grid_points[:,0,:]
    for i in range(nradius):
        f = float(i)/(nradius-1)
        r = radius0 + f*(radius1 - radius0)
        multiply(grid_points[:,i,:], r, grid_points[:,i,:])
    from chimerax.geometry import translation, orthonormal_frame
    tf = translation(center) * orthonormal_frame(axis)
    tf.transform_points(grid_points.reshape((ncircum*nradius,3)), in_place = True)
    return grid_points

# -----------------------------------------------------------------------------
#
def cylinder_radii(v, center, axis):

    rmins = []
    rmaxs = []
    from numpy import empty, single as floatc
    for p in v.surfaces:
        vertices = p.vertices
        r = empty((len(vertices),), floatc)
        from chimerax.geometry.vector import distances_perpendicular_to_axis
        distances_perpendicular_to_axis(vertices, center, axis, r)
        rmins.append(r.min())
        rmaxs.append(r.max())
    rmin, rmax = min(rmins), max(rmaxs)
    return rmin, rmax
