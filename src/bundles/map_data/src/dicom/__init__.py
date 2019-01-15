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
# DICOM map file reader.
#
def open(paths, log = None, verbose = False):

  if isinstance(paths, str):
    paths = [paths]
  from .dicom_grid import dicom_grids
  return dicom_grids(paths, log = log, verbose = verbose)
