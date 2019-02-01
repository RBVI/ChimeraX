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
# Chimera HDF map file reader.
#
def open(path):

  from .cmap_grid import read_chimera_map
  return read_chimera_map(path)

# -----------------------------------------------------------------------------
#
from .cmap_format import copy_hdf5_array
from .write_cmap import write_grid_as_chimera_map as save
