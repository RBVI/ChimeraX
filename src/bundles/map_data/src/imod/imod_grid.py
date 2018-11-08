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
# Wrap CCP4 image data as grid data for displaying surface, meshes, and volumes.
#
from ..mrc.mrc_grid import MRCGrid

# -----------------------------------------------------------------------------
#
class IMODGrid(MRCGrid):

  def __init__(self, path):
    MRCGrid.__init__(self, path, file_type = 'imod')
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    d = MRCGrid.read_matrix(self, ijk_origin, ijk_size, ijk_step, progress)

    import numpy
    if self.value_type == numpy.uint8:
      # Invert 8-bit unsigned map.  Most commonly this is tomography data
      # with low map values corresponding to high density values.
      numpy.subtract(255, d, d)

    return d
