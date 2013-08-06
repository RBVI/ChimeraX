# -----------------------------------------------------------------------------
# DSN6 and BRIX electron density map file reader.
# Used by crystallography visualization program O, usual file suffix .omap.
#
def open(path):

  from .dsn6_grid import DSN6_Grid
  return [DSN6_Grid(path)]

# -----------------------------------------------------------------------------
#
from .writebrix import write_brix
