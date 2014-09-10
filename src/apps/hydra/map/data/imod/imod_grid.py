# -----------------------------------------------------------------------------
# Wrap CCP4 image data as grid data for displaying surface, meshes, and volumes.
#
from ..mrc.mrc_grid import MRC_Grid

# -----------------------------------------------------------------------------
#
class IMOD_Grid(MRC_Grid):

  def __init__(self, path):
    MRC_Grid.__init__(self, path, file_type = 'imod')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    d = MRC_Grid.read_matrix(self, ijk_origin, ijk_size, ijk_step, progress)

    import numpy
    if self.value_type == numpy.uint8:
      # Invert 8-bit unsigned map.  Most commonly this is tomography data
      # with low map values corresponding to high density values.
      numpy.subtract(255, d, d)

    return d
