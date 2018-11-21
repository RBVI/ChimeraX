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
# Wrap MacMolPlt "3D surface" format used for quantum mechanical density and
# electrostatics calculations with Gamess.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class MacMolPltGrid(GridData):

  def __init__(self, path):

    from . import macmolplt_format
    md = macmolplt_format.MacMolPlt_Data(path)
    self.macmolplt_data = md

    GridData.__init__(self, md.grid_size,
                      origin = md.origin, step = md.step,
                      path = path, file_type = 'macmolplt')

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    matrix = self.macmolplt_data.matrix(progress)
    if ijk_size != self.size:
      self.cache_data(matrix, (0,0,0), self.size, (1,1,1)) # Cache full data.
    m = self.matrix_slice(matrix, ijk_origin, ijk_size, ijk_step)
    return m
