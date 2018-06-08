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
# NetCDF array file reader.
# Used as a generic scientific array data format.
#

from .netcdf_grid import read_netcdf
from .netcdf_format import write_grid_as_netcdf

# -----------------------------------------------------------------------------
#
def open(path):

  return read_netcdf(path)
