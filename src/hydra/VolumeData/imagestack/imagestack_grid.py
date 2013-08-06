# -----------------------------------------------------------------------------
# Wrap image data as grid data for displaying surface, meshes, and volumes.
#
from VolumeData import Grid_Data

# -----------------------------------------------------------------------------
#
class Image_Stack_Grid(Grid_Data):

  def __init__(self, paths):

    import imagestack_format
    d = imagestack_format.Image_Stack_Data(paths)
    self.image_stack = d

    Grid_Data.__init__(self, d.data_size, d.value_type,
                       d.data_origin, d.data_step,
                       path = d.paths, file_type = 'imagestack')

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from VolumeData.readarray import allocate_array
    m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    self.image_stack.read_matrix(ijk_origin, ijk_size, ijk_step, m, progress)
    return m
