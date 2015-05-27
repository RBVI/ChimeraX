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
        volume.show()
        fv = volume
    else:
        fg = falloff_grid(volume, iterations, step, subregion)
        from .. import volume_from_grid_data
        fv = volume_from_grid_data(fg, show_data = False, model_id = modelId,
                                   session = session)
        fv.copy_settings_from(volume)
        fv.show()
        volume.unshow()          # Hide original map
  
    return fv

# -----------------------------------------------------------------------------
#
def falloff_grid(volume, iterations = 10, step = 1, subregion = None):

    v = volume
    region = v.subregion(step, subregion)

    m = v.region_matrix(region).copy()
    falloff_matrix(m, iterations)

    from ..data import Array_Grid_Data
    d = v.data
    name = '%s falloff' % v.name
    forigin, fstep = v.region_origin_and_step(region)
    fg = Array_Grid_Data(m, forigin, fstep, d.cell_angles, d.rotation,
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
