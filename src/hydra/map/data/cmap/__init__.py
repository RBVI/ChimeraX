# -----------------------------------------------------------------------------
# Chimera HDF map file reader.
#
def open(path):

  from .cmap_grid import read_chimera_map
  return read_chimera_map(path)

# -----------------------------------------------------------------------------
#
from .write_cmap import write_grid_as_chimera_map
