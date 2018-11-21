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
# Wrap SITUS density maps as grid data for displaying surface, meshes,
# and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class SITUSGrid(GridData):

  def __init__(self, path):

    from . import situs_format
    sm = situs_format.SITUS_Density_Map(path)
    self.density_map = sm

    step = (sm.voxel_size, sm.voxel_size, sm.voxel_size)

    GridData.__init__(self, sm.grid_size, origin = sm.origin, step = step,
                      path = path, file_type = 'situs')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.density_map.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
