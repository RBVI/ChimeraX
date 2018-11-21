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
# Wrap DelPhi or GRASP unformatted electrostatic potential file
# (usual extension .phi) as grid data for displaying surface, meshes,
# and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class DelPhiGrid(GridData):

  def __init__(self, path):

    from . import delphi_format
    dd = delphi_format.DelPhi_Data(path)
    self.delphi_data = dd

    GridData.__init__(self, dd.size, dd.value_type,
                      origin = dd.xyz_origin, step = dd.xyz_step,
                      path = path, file_type = 'delphi')

    self.polar_values = True
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.delphi_data.matrix(ijk_origin, ijk_size, ijk_step, progress)
    return matrix
