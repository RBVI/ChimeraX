# -----------------------------------------------------------------------------
# Wrap EM density map data as grid data for displaying surface, meshes, and
# volumes.
#
from VolumeData import Grid_Data

# -----------------------------------------------------------------------------
#
class EM_Grid(Grid_Data):

  def __init__(self, path):

    import em_format
    d = em_format.EM_Data(path)
    self.em_data = d

    Grid_Data.__init__(self, d.data_size, d.element_type,
                       d.data_origin, d.data_step,
                       path = path, file_type = 'tom_em')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.em_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
    return matrix
