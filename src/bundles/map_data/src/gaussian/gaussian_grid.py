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
# Wrap Gaussian energy maps as grid data for displaying surface, meshes,
# and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class GaussianGrid(GridData):

  def __init__(self, gc, component_number):

    self.gc = gc
    self.component_number = component_number

    from chimerax.geometry import matrix
    ca, rot = matrix.cell_angles_and_rotation(gc.grid_axes)

    GridData.__init__(self, gc.grid_size,
                      origin = gc.origin, step = gc.step,
                      cell_angles = ca, rotation = rot,
                      path = gc.path, file_type = 'gaussian',
                      grid_id = str(component_number))

    self.polar_values = True
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.gc.matrix(self.component_number, progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m

# -----------------------------------------------------------------------------
#
def read_gaussian_file(path):

    from . import gaussian_format
    gc = gaussian_format.Gaussian_Cube(path)

    grids = []
    for c in range(gc.num_components):
      g = GaussianGrid(gc, c)
      grids.append(g)

    return grids
