# -----------------------------------------------------------------------------
# plt file reader.
# Used by visualization program gOpenMol, usual file suffix .plt.
#
def open(path):

  from plt_grid import Plt_Grid
  return [Plt_Grid(path)]
