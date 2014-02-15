# -----------------------------------------------------------------------------
# Wrap UHBD data as grid data for displaying surface, meshes, and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class UHBD_Grid(Grid_Data):

  def __init__(self, path):

    import uhbd_format
    d = uhbd_format.UHBD_Data(path)
    self.uhbd_data = d

    Grid_Data.__init__(self, d.data_size,
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
