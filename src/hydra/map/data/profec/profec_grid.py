# -----------------------------------------------------------------------------
# Wrap PROFEC energy maps as grid data for displaying surface, meshes,
# and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class PROFEC_Grid(Grid_Data):

  def __init__(self, path):

    import profec_format
    eg = profec_format.PROFEC_Potential(path)
    self.energy_grid = eg
    import Matrix
    r = Matrix.orthogonalize(eg.rotation)
    Grid_Data.__init__(self, eg.grid_size,
                       origin = eg.origin, step = eg.step,
                       rotation = r,
                       path = path, file_type = 'profec')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.energy_grid.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
