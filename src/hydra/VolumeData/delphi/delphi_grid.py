# -----------------------------------------------------------------------------
# Wrap DelPhi or GRASP unformatted electrostatic potential file
# (usual extension .phi) as grid data for displaying surface, meshes,
# and volumes.
#
from VolumeData import Grid_Data

# -----------------------------------------------------------------------------
#
class DelPhi_Grid(Grid_Data):

  def __init__(self, path):

    import delphi_format
    dd = delphi_format.DelPhi_Data(path)
    self.delphi_data = dd

    Grid_Data.__init__(self, dd.size, dd.value_type,
                       origin = dd.xyz_origin, step = dd.xyz_step,
                       path = path, file_type = 'delphi')

    self.polar_values = True
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.delphi_data.matrix(ijk_origin, ijk_size, ijk_step, progress)
    return matrix
