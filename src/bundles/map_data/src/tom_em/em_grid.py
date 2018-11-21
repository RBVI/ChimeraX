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
# Wrap EM density map data as grid data for displaying surface, meshes, and
# volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class EMGrid(GridData):

  def __init__(self, path):

    from . import em_format
    d = em_format.EM_Data(path)
    self.em_data = d

    GridData.__init__(self, d.data_size, d.element_type,
                      d.data_origin, d.data_step,
                      path = path, file_type = 'tom_em')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.em_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
    return matrix
