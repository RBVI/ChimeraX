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
# Wrap plt maps (used by gOpenMol, usual extension .plt)
# as grid data for displaying surface, meshes, and volumes.
#
from .. import GridData

class PltGrid(GridData):

	def __init__(self, path):
		from . import plt_format
		dm = plt_format.Plt_map(path)
		self.density_map = dm

		size = dm.extent

		GridData.__init__(self, size,
			          origin = dm.origin, step = dm.grid,
			          path = path, file_type = 'gopenmol')

		self.polar_values = True

	def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
		data = self.density_map.matrix(ijk_origin, ijk_size, ijk_step,
					       progress)
		return data
