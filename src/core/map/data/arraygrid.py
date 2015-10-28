# vim: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
# Array_Grid_Data objects wraps a NumPy array for use by volume viewer.
#
from . import Grid_Data

# -----------------------------------------------------------------------------
# Constructor requires NumPy arrays with indices in z, y, x order.
# Origin and step parameters are in x, y, z order.
#
class Array_Grid_Data(Grid_Data):
  
  def __init__(self, array, origin = (0,0,0), step = (1,1,1),
               cell_angles = (90,90,90),
               rotation = ((1,0,0),(0,1,0),(0,0,1)),
               symmetries = (),
               name = ''):

      self.array = array
      
      path = ''
      file_type = ''
      component_name = ''

      grid_size = list(array.shape)
      grid_size.reverse()

      value_type = array.dtype

      Grid_Data.__init__(self, grid_size, value_type,
                         origin, step, cell_angles = cell_angles,
                         rotation = rotation, symmetries = symmetries,
                         name = name, path = path, file_type = file_type)

      self.writable = True
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
  
    return self.cached_data(ijk_origin, ijk_size, ijk_step)

  # ---------------------------------------------------------------------------
  #
  def cached_data(self, ijk_origin, ijk_size, ijk_step):

      m = self.matrix_slice(self.array, ijk_origin, ijk_size, ijk_step)
      return m
