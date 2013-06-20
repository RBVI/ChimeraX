# -----------------------------------------------------------------------------
# Situs density map file reader.
#
def open(path):

  from situs_grid import SITUS_Grid
  return [SITUS_Grid(path)]
