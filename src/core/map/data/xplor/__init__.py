# -----------------------------------------------------------------------------
# XPLOR unformatted ascii density map file reader.
#
def open(path):

  from xplor_grid import XPLOR_Grid
  return [XPLOR_Grid(path)]
