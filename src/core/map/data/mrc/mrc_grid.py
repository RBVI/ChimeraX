# -----------------------------------------------------------------------------
# Wrap MRC image data as grid data for displaying surface, meshes, and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class MRC_Grid(Grid_Data):

  def __init__(self, path, file_type = 'mrc'):

    from . import mrc_format
    d = mrc_format.MRC_Data(path, file_type)

    self.mrc_data = d

    Grid_Data.__init__(self, d.data_size, d.element_type,
                       d.data_origin, d.data_step, d.cell_angles, d.rotation,
                       path = path, file_type = file_type)

    self.unit_cell_size = d.unit_cell_size

    # Crystal symmetry operators.
#    syms = d.symmetry_matrices()
#    if syms:
#      self.symmetries = syms
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    return self.mrc_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)

  # ---------------------------------------------------------------------------
  # MRC format does not support unsigned 8-bit integers although it is
  # sometimes used for such data with data type incorrectly specified as
  # signed 8-bit integers.
  #
  def signed8_to_unsigned8(self):

    from numpy import int8, uint8, dtype
    if self.mrc_data.element_type == int8:
      self.mrc_data.element_type = uint8
      self.value_type = dtype(uint8)
      self.clear_cache()
      self.values_changed()
