# -----------------------------------------------------------------------------
# Compute a map that gives the local correlation between two maps.
# At each grid point the correlation over a box N by N by N voxels in size
# centered at that point is computed.  The computation is done on the grid
# of the first map, and the second map is interpolated at those grid points
# if its grid does not align.
#
# This can be used for coloring # surfaces as shown in figure S3 of Hipp et al.
# Nucleic Acids Research, 2012, Vol. 40, No. 7 3275-3288
#
def local_correlation(map1, map2, window_size, subtract_mean, model_id = None):

    d1 = map1.data
    m1 = map1.full_matrix()
    d2 = map2.data
    m2, same = map2.interpolate_on_grid(map1)
    mc = local_correlation_matrix(m1, m2, window_size, subtract_mean)

    hs = 0.5*(window_size-1)
    origin = tuple(o+hs*s for o,s in zip(d1.origin, d1.step))
    from VolumeData import Array_Grid_Data
    g = Array_Grid_Data(mc, origin, d1.step, d1.cell_angles, d2.rotation,
                        name = 'local correlation')

    from VolumeViewer import volume_from_grid_data
    mapc = volume_from_grid_data(g, model_id = model_id)
    mapc.openState.xform = map1.openState.xform

    return mapc

# -----------------------------------------------------------------------------
#
def local_correlation_matrix(m1, m2, window_size, subtract_mean):

    w = window_size
    shape = tuple(s-w+1 for s in m1.shape)
    from numpy import empty, float32
    mc = empty(shape, float32)

    if m1.dtype != m2.dtype:
        # _volume routine requires both matrices have same value type.
        m1 = m1.astype(float32)
        m2 = m2.astype(float32)

    import _volume
    _volume.local_correlation(m1, m2, window_size, subtract_mean, mc)
    return mc

# -----------------------------------------------------------------------------
# Slow.  Do this in C++ instead.
#
def local_correlation_calc(m1, m2, window_size, mc):

    w = window_size
    from math import sqrt
    ksize, jsize, isize = shape
    for k in range(ksize):
        for j in range(jsize):
            for i in range(isize):
                ml1 = m1[k:k+w,j:j+w,i:i+w].astype(float32)
                ml2 = m2[k:k+w,j:j+w,i:i+w].astype(float32)
                s12 = (ml1*ml2).sum()
                s1 = sqrt((ml1*ml1).sum())
                s2 = sqrt((ml2*ml2).sum())
                s1s2 = s1*s2
                mc[k,j,i] = s12/s1s2 if s1s2 != 0 else 0

    return mc
