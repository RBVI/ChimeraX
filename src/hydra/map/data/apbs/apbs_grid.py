# -----------------------------------------------------------------------------
# Wrap Adaptive Poisson-Boltzmann Solver (APBS) electrostatics
# opendx file as grid data for displaying surface, meshes,
# and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class APBS_Grid(Grid_Data):

  def __init__(self, path):

    import apbs_format
    ad = apbs_format.APBS_Data(path)
    self.apbs_data = ad

    Grid_Data.__init__(self, ad.grid_size,
                       origin = ad.xyz_origin, step = ad.xyz_step,
                       path = path, file_type = 'apbs')

    self.polar_values = True

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.apbs_data.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
