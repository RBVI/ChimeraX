# -----------------------------------------------------------------------------
# Purdue Image Format density map file reader.
# Used by Tim Baker's lab at UC San Diego for electron microscope density
# files.  Used with ROBEM visualization program.
#
def open(path):

  from pif_grid import PIF_Grid
  return [PIF_Grid(path)]
