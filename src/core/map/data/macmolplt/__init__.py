# -----------------------------------------------------------------------------
# MacMolPlt molecular orbital and electrostatic potential data.
#
def open(path):

  from macmolplt_grid import MacMolPlt_Grid
  return [MacMolPlt_Grid(path)]
