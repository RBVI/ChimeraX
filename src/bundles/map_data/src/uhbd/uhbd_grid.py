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
# Wrap UHBD data as grid data for displaying surface, meshes, and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class UHBDGrid(GridData):

  def __init__(self, path):

    from . import uhbd_format
    d = uhbd_format.UHBD_Data(path)
    self.uhbd_data = d

    GridData.__init__(self, d.data_size,
                      origin = d.data_origin, step = d.data_step,
                      path = path, file_type = 'uhbd')

    self.polar_values = True
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.uhbd_data.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
