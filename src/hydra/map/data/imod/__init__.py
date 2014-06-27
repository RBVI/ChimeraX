# -----------------------------------------------------------------------------
# IMOD mrc density map file reader.  IMOD uses mrc signed 8-bit mode as
# unsigned.
#
def open(path):

  from imod_grid import IMOD_Grid
  return [IMOD_Grid(path)]
