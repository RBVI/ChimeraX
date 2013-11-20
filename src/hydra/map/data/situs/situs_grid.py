# -----------------------------------------------------------------------------
# Wrap SITUS density maps as grid data for displaying surface, meshes,
# and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class SITUS_Grid(Grid_Data):

  def __init__(self, path):

    import situs_format
    sm = situs_format.SITUS_Density_Map(path)
    self.density_map = sm

    step = (sm.voxel_size, sm.voxel_size, sm.voxel_size)

    Grid_Data.__init__(self, sm.grid_size, origin = sm.origin, step = step,
                       path = path, file_type = 'situs')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.density_map.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
