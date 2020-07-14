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
# Smooth edge of masked volume by replacing values outside density by average
# of 6 neighbor grid values, repeating a specified number of iterations.
# "Outside" grid points are defined to be those with value 0.
#
def falloff(volume, iterations = 10, in_place = False,
            step = 1, subregion = None, modelId = None, session = None):

    if in_place:
        if not volume.data.writable:
            raise ValueError("Can't modify read-only volume %s in place"
                             % volume.name)
        falloff_matrix(volume.full_matrix(), iterations)
        volume.matrix_changed()
        fv = volume
    else:
        fg = falloff_grid(volume, iterations, step, subregion)
        from chimerax.map import volume_from_grid_data
        fv = volume_from_grid_data(fg, model_id = modelId, session = session)
        fv.copy_settings_from(volume)
        volume.display = False          # Hide original map
  
    return fv

# -----------------------------------------------------------------------------
#
def falloff_grid(volume, iterations = 10, step = 1, subregion = None):

    v = volume
    region = v.subregion(step, subregion)

    from numpy import float32
    m = v.region_matrix(region).astype(float32)
    falloff_matrix(m, iterations)

    from chimerax.map_data import ArrayGridData
    d = v.data
    name = '%s falloff' % v.name
    forigin, fstep = v.region_origin_and_step(region)
    fg = ArrayGridData(m, forigin, fstep, d.cell_angles, d.rotation,
                       name = name)
    return fg

# -----------------------------------------------------------------------------
#
def falloff_matrix(m, iterations = 10):

    from numpy import empty, putmask
    mask = (m == 0)
    ave = empty(m.shape, m.dtype)
    for s in range(iterations):
        ave[:,:,:] = 0
        ave[:-1,:,:] += m[1:,:,:]
        ave[1:,:,:] += m[:-1,:,:]
        ave[:,:-1,:] += m[:,1:,:]
        ave[:,1:,:] += m[:,:-1,:]
        ave[:,:,:-1] += m[:,:,1:]
        ave[:,:,1:] += m[:,:,:-1]
        ave /= 6
        putmask(m, mask, ave)
