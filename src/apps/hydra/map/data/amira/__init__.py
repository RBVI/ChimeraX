# -----------------------------------------------------------------------------
# AmiraMesh map file reader.
#
def open(path):

  from amira_grid import Amira_Grid
  return [Amira_Grid(path)]
