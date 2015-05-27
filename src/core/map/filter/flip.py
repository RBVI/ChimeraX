# -----------------------------------------------------------------------------
# Reverse order of planes along x, y or z axes.
# Origin and grid spacing remain the same.
#
# This is not the same as inverting the z axis if cell angles are not
# 90 degrees.  Does not change rotation or symmetries.
#
from VolumeData import Grid_Data
class Flip_Grid(Grid_Data):

    def __init__(self, grid_data, axes = 'z'):

        d = grid_data
        self.data = d
        self.axes = axes

        Grid_Data.__init__(self, d.size, d.value_type,
                           d.origin, d.step, d.cell_angles,
                           d.rotation, d.symmetries,
                           name = '%s %s flip' % (d.name, axes),
                           file_type = d.file_type,
                           default_color = d.rgba)
        self.data_cache = None      # Caching done by underlying grid.
        
    # -------------------------------------------------------------------------
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

        origin = self.flipped_origin(ijk_origin, ijk_size)
        m = self.data.matrix(origin, ijk_size, ijk_step, progress)
        mf = flip_matrix(m, self.axes)
        return mf
        
    # -------------------------------------------------------------------------
    #
    def cached_data(self, ijk_origin, ijk_size, ijk_step):

        origin = self.flipped_origin(ijk_origin, ijk_size)
        m = self.data.cached_data(origin, ijk_size, ijk_step)
        if m is None:
            return m
        mf = flip_matrix(m, self.axes)
        return mf
        
    # -------------------------------------------------------------------------
    #
    def flipped_origin(self, ijk_origin, ijk_size):

        origin = list(ijk_origin)
        axis = {'x':0, 'y':1, 'z':2}
        for a in self.axes:
            ai = axis[a]
            origin[ai] = self.data.size[ai] - (ijk_origin[ai] + ijk_size[ai])
        return origin

    # -------------------------------------------------------------------------
    #
    def clear_cache(self):

        self.data.clear_cache()

# -----------------------------------------------------------------------------
#
def flip_matrix(m, axes):

    for a in axes:
        if a == 'z':
            m = m[::-1,:,:]
        elif a == 'y':
            m = m[:,::-1,:]
        elif a == 'x':
            m = m[:,:,::-1]
    return m

# -----------------------------------------------------------------------------
#
def flip_in_place(m, axes):

    if 'z' in axes:
        n = m.shape[0]
        for k in range(n/2-1):
            p = m[k,:,:].copy()
            m[k,:,:] = m[n-1-k,:,:]
            m[n-1-k,:,:] = p
    if 'y' in axes:
        n = m.shape[1]
        for k in range(n/2-1):
            p = m[:,k,:].copy()
            m[:,k,:] = m[:,n-1-k,:]
            m[:,n-1-k,:] = p
    if 'x' in axes:
        n = m.shape[2]
        for k in range(n/2-1):
            p = m[:,:,k].copy()
            m[:,:,k] = m[:,:,n-1-k]
            m[:,:,n-1-k] = p
