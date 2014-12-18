# -----------------------------------------------------------------------------
# Compute center of mass of a map for the region above a specifie contour level.
#
def volume_center_of_mass(v, level = None, step = None, subregion = None):

    if level is None:
        # Use lowest displayed contour level.
        level = min(v.surface_levels)

    # Get 3-d array of map values.
    m = v.matrix(step = step, subregion = subregion)

    # Find indices of map values above displayed threshold.
    kji = (m >= level).nonzero()

    # Compute total mass above threshold.
    msum = m[kji].sum()

    # Compute mass-weighted center
    mcenter = [(i*m[kji]).sum()/msum for i in kji]
    mcenter.reverse()        # k,j,i -> i,j,k index order

    tf = v.matrix_indices_to_xyz_transform(step, subregion)
    center = tf*mcenter

    return center
