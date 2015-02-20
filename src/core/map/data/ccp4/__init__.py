# -----------------------------------------------------------------------------
# CCP4 density map file reader.
#
def open(path):

  from .ccp4_grid import CCP4_Grid
  return [CCP4_Grid(path)]
