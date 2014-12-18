# -----------------------------------------------------------------------------
# Wrap PIF image data as grid data for displaying surface, meshes, and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class PIF_Grid(Grid_Data):

  def __init__(self, path):

    import pif_format
    d = pif_format.PIF_Data(path)
    self.pif_data = d

    Grid_Data.__init__(self, d.data_size, d.element_type,
                       d.data_origin, d.data_step, d.cell_angles,
                       path = path, file_type = 'pif')

    self.polar_values = True    # These maps are frequently inverted.
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    return self.pif_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
