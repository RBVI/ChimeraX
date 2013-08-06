# -----------------------------------------------------------------------------
# Wrap CCP4 image data as grid data for displaying surface, meshes, and volumes.
#
from ..mrc.mrc_grid import MRC_Grid

# -----------------------------------------------------------------------------
#
class CCP4_Grid(MRC_Grid):
  def __init__(self, path):
    MRC_Grid.__init__(self, path, file_type = 'ccp4')
