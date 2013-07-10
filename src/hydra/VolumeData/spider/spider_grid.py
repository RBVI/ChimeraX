# -----------------------------------------------------------------------------
# Wrap SPIDER volume data as grid data for displaying surface, meshes, and
# volumes.
#
from VolumeData import Grid_Data

# -----------------------------------------------------------------------------
#
class SPIDER_Grid(Grid_Data):

  def __init__(self, path):

    import spider_format
    d = spider_format.SPIDER_Data(path)
    self.spider_data = d

    origin = map(lambda a, b: a * b, d.data_origin, d.data_step)

    Grid_Data.__init__(self, d.data_size, origin = origin, step = d.data_step,
                       path = path, file_type = 'spider')

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    return self.spider_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                                        progress)
