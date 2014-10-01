# -----------------------------------------------------------------------------
# Compute center of mass of a map for the region above a specifie contour level.
# Returns center map index coordinates.
#
def volume_center_of_mass(v, level = None):

    if level is None:
        # Use lowest displayed contour level.
        level = min(v.surface_levels)

    # Get 3-d array of map values.
    m = v.data.full_matrix()

    # Find indices of map values above displayed threshold.
    kji = (m >= level).nonzero()

    # Compute total mass above threshold.
    msum = m[kji].sum()

    # Compute mass-weighted center
    center = [(i*m[kji]).sum()/msum for i in kji]
    center.reverse()        # k,j,i -> i,j,k index order

    return center
