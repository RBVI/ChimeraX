# -----------------------------------------------------------------------------
# File reader for Adaptive Poisson-Boltzmann Solver (APBS) electrostatics
# opendx file.
#
def open(path):

  from .apbs_grid import APBS_Grid
  return [APBS_Grid(path)]
