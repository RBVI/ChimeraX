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
# DSN6 and BRIX electron density map file reader.
# Used by crystallography visualization program O, usual file suffix .omap.
#
def open(path):

  from .dsn6_grid import DSN6Grid
  return [DSN6Grid(path)]

# -----------------------------------------------------------------------------
#
from .writebrix import write_brix as save
