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
# Wrap AmiraMesh data as grid data for displaying surface, meshes, and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class AmiraGrid(GridData):

  def __init__(self, path):

    from . import amira_format
    d = amira_format.Amira_Mesh_Data(path)

    self.amira_data = d

    GridData.__init__(self, d.matrix_size, d.element_type, step = d.step,
                      path = path, file_type = 'amira')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    return self.amira_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
