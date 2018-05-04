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

# Make a new map by squaring the values of the currently active map.

# Get the currently active volume
from VolumeViewer import active_volume
v = active_volume()

# Square the density values.
import numpy
m = v.full_matrix().astype(numpy.float32)
m[:] = m*m

# Make a new map.
from VolumeData import Array_Grid_Data
from VolumeViewer.volume import volume_from_grid_data
g = Array_Grid_Data(m, v.data.origin, v.data.step, v.data.cell_angles)
c = volume_from_grid_data(g)
