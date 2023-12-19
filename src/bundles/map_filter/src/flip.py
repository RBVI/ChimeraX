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
# Reverse order of planes along x, y or z axes.
# Origin and grid spacing remain the same.
#
# This is not the same as inverting the z axis if cell angles are not
# 90 degrees.  Does not change rotation or symmetries.
#
from chimerax.map_data import GridData
class FlipGrid(GridData):

    def __init__(self, grid_data, axes = 'z'):

        d = grid_data
        self.data = d
        self.axes = axes

        settings = d.settings(name = '%s %s flip' % (d.name, axes))
        GridData.__init__(self, file_type = d.file_type, **settings)
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
        for k in range(n//2-1):
            p = m[k,:,:].copy()
            m[k,:,:] = m[n-1-k,:,:]
            m[n-1-k,:,:] = p
    if 'y' in axes:
        n = m.shape[1]
        for k in range(n//2-1):
            p = m[:,k,:].copy()
            m[:,k,:] = m[:,n-1-k,:]
            m[:,n-1-k,:] = p
    if 'x' in axes:
        n = m.shape[2]
        for k in range(n//2-1):
            p = m[:,:,k].copy()
            m[:,:,k] = m[:,:,n-1-k]
            m[:,:,n-1-k] = p
