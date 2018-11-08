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
# Wrap XPLOR density maps as grid data for displaying surface, meshes,
# and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class XPLORGrid(GridData):

  def __init__(self, path):

    from . import xplor_format
    xm = xplor_format.XPLOR_Density_Map(path)
    self.density_map = xm

    step =  tuple(cs / gs for cs,gs in zip(xm.cell_size, (xm.na, xm.nb, xm.nc)))
    from ..griddata import scale_and_skew
    origin = scale_and_skew((xm.amin, xm.bmin, xm.cmin), step, xm.cell_angles)

    GridData.__init__(self, xm.grid_size, origin = origin, step = step,
                      cell_angles = xm.cell_angles,
                      path = path, file_type = 'xplor')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress = None):

    matrix = self.density_map.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
