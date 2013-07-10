# -----------------------------------------------------------------------------
# File reader for University of Houston Brownian Dynamics (UHBD) grids.
# Used to compute electrostatic potential, and other properties.
#
def open(path):

  from uhbd_grid import UHBD_Grid
  return [UHBD_Grid(path)]
