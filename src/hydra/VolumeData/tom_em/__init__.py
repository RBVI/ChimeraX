# -----------------------------------------------------------------------------
# TOM Toolbox EM density map file reader (http://www.biochem.mpg.de/tom/).
#
def open(path):

  from em_grid import EM_Grid
  return [EM_Grid(path)]
