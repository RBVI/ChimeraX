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
# MRC density map file reader.
# Used by Wah Chiu's group at Baylor School of Medicine for electron microscope
# density files.
#
from .writemrc import write_mrc2000_grid_data as save

# -----------------------------------------------------------------------------
#
def open(path):

  from .mrc_grid import MRCGrid
  return [MRCGrid(path)]
