# -----------------------------------------------------------------------------
# Wrap MacMolPlt "3D surface" format used for quantum mechanical density and
# electrostatics calculations with Gamess.
#
from VolumeData import Grid_Data

# -----------------------------------------------------------------------------
#
class MacMolPlt_Grid(Grid_Data):

  def __init__(self, path):

    import macmolplt_format
    md = macmolplt_format.MacMolPlt_Data(path)
    self.macmolplt_data = md

    Grid_Data.__init__(self, md.grid_size,
                       origin = md.origin, step = md.step,
                       path = path, file_type = 'macmolplt')

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.macmolplt_data.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
