# -----------------------------------------------------------------------------
# Wrap XPLOR density maps as grid data for displaying surface, meshes,
# and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class XPLOR_Grid(Grid_Data):

  def __init__(self, path):

    import xplor_format
    xm = xplor_format.XPLOR_Density_Map(path)
    self.density_map = xm

    step = map(lambda cs, gs: cs / gs, xm.cell_size, (xm.na, xm.nb, xm.nc))
    from ..griddata import scale_and_skew
    origin = scale_and_skew((xm.amin, xm.bmin, xm.cmin), step, xm.cell_angles)

    Grid_Data.__init__(self, xm.grid_size, origin = origin, step = step,
                       cell_angles = xm.cell_angles,
                       path = path, file_type = 'xplor')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress = None):

    matrix = self.density_map.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
