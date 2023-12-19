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

from ..mrc.writemrc import write_ccp4_grid_data as save

# -----------------------------------------------------------------------------
# CCP4 density map file reader.
#
def open(path):

  from .ccp4_grid import CCP4Grid
  return [CCP4Grid(path)]
