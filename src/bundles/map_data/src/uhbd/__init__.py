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
# File reader for University of Houston Brownian Dynamics (UHBD) grids.
# Used to compute electrostatic potential, and other properties.
#
def open(path):

  from .uhbd_grid import UHBDGrid
  return [UHBDGrid(path)]
