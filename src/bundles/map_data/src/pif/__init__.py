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
# Purdue Image Format density map file reader.
# Used by Tim Baker's lab at UC San Diego for electron microscope density
# files.  Used with ROBEM visualization program.
#
def open(path):

  from .pif_grid import PIFGrid
  return [PIFGrid(path)]
