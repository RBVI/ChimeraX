# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Wrap MRC image data as grid data for displaying surface, meshes, and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class MRCGrid(GridData):

  def __init__(self, path, file_type = 'mrc'):

    from . import mrc_format
    d = mrc_format.MRC_Data(path, file_type)

    self.mrc_data = d

    # Read float16 as float32 since C++ routines can't handle float16.
    element_type = d.element_type
    from numpy import float16, float32
    if element_type == float16:
      element_type = float32
      
    GridData.__init__(self, d.data_size, element_type,
                      d.data_origin, d.data_step, d.cell_angles, d.rotation,
                      path = path, file_type = file_type)

    self.unit_cell_size = d.unit_cell_size
    self.file_header = d.header

    # Crystal symmetry operators.
    syms = d.symmetry_matrices()
    if syms:
      self.symmetries = syms
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    m = self.mrc_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)

    from numpy import float16, float32
    if m.dtype == float16:
      m = m.astype(float32)

    return m

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
