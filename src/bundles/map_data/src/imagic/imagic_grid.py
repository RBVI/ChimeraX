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

def imagic_grids(path, file_type = 'hed'):
    from .imagic_format import IMAGIC_Data
    h = IMAGIC_Data(path, file_type)
    grids = [IMAGICGrid(path, h, series_index) for series_index in range(h.num_volumes)]
    return grids
  
# -----------------------------------------------------------------------------
# Wrap IMAGIC image data as grid data for displaying surface, meshes, and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class IMAGICGrid(GridData):

  def __init__(self, path, imagic_data, series_index):

    self.imagic_data = h = imagic_data

    GridData.__init__(self, h.data_size, h.element_type,
                      h.data_origin, h.data_step, h.cell_angles, h.rotation,
                      path = path, file_type = h.file_type, time = series_index)

    self.unit_cell_size = h.unit_cell_size
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    return self.imagic_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress,
                                        series_index = self.time)
