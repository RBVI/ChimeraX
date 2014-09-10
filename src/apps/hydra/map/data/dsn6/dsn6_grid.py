# -----------------------------------------------------------------------------
# Wrap DSN6 and BRIX electron density maps (used by crystallography program O,
# usual extension .omap) as grid data for displaying surface, meshes, and
# volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class DSN6_Grid(Grid_Data):

  def __init__(self, path):

    from . import dsn6_format
    dm = dsn6_format.dsn6_map(path)
    self.density_map = dm

    size = dm.extent
    step = dm.cell[:3] / dm.grid
    cell_angles = tuple(dm.cell[3:])
    from ..griddata import scale_and_skew
    origin = scale_and_skew(dm.origin, step, cell_angles)
    from numpy import float32

    Grid_Data.__init__(self, size, float32, origin, step, cell_angles,
                       path = path, file_type = 'dsn6')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    data = self.density_map.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(data, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(data, ijk_origin, ijk_size, ijk_step)
    return m
  
  # ---------------------------------------------------------------------------
  # Chimera 1.2467 and earlier did not use dsn6 prod/plus value scaling.
  #
  def use_value_scaling(self, use):
    self.density_map.scale_values = use
  def using_value_scaling(self):
    return self.density_map.scale_values
