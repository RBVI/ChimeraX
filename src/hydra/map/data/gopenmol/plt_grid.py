# -----------------------------------------------------------------------------
# Wrap plt maps (used by gOpenMol, usual extension .plt)
# as grid data for displaying surface, meshes, and volumes.
#
from VolumeData import Grid_Data

class Plt_Grid(Grid_Data):

	def __init__(self, path):
		import plt_format
		dm = plt_format.Plt_map(path)
		self.density_map = dm

		size = dm.extent

		Grid_Data.__init__(self, size,
				   origin = dm.origin, step = dm.grid,
				   path = path, file_type = 'gopenmol')

		self.polar_values = True

	def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
		data = self.density_map.matrix(ijk_origin, ijk_size, ijk_step,
					       progress)
		return data
