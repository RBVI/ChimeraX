# -----------------------------------------------------------------------------
# MRC density map file reader.
# Used by Wah Chiu's group at Baylor School of Medicine for electron microscope
# density files.
#
from .writemrc import write_mrc2000_grid_data

# -----------------------------------------------------------------------------
#
def open(path):

  from .mrc_grid import MRC_Grid
  return [MRC_Grid(path)]
