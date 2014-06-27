# -----------------------------------------------------------------------------
# DelPhi electrostatic potential file reader.
# Usual file suffix .phi
#
def open(path):

  from delphi_grid import DelPhi_Grid
  return [DelPhi_Grid(path)]
