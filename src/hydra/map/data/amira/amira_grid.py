# -----------------------------------------------------------------------------
# Wrap AmiraMesh data as grid data for displaying surface, meshes, and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class Amira_Grid(Grid_Data):

  def __init__(self, path):

    import amira_format
    d = amira_format.Amira_Mesh_Data(path)

    self.amira_data = d

    Grid_Data.__init__(self, d.matrix_size, d.element_type, step = d.step,
                       path = path, file_type = 'amira')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    return self.amira_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
