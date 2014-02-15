# -----------------------------------------------------------------------------
# Wrap EMAN HDF image data as grid data for displaying surface, meshes,
#  and volumes.
#
from .. import Grid_Data

# -----------------------------------------------------------------------------
#
class EMAN_HDF_Grid(Grid_Data):

  def __init__(self, eman_hdf_data, size, value_type,
               origin, step, array_path):

    self.eman_hdf_data = eman_hdf_data
    self.array_path = array_path

    Grid_Data.__init__(self, size, value_type,
                       origin, step,
                       path = eman_hdf_data.path, file_type = 'emanhdf',
                       grid_id = array_path)
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import allocate_array
    m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    self.eman_hdf_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                                   self.array_path, m, progress)
    return m
    
# -----------------------------------------------------------------------------
#
def read_eman_hdf5(path):

  from eman_hdf_format import EMAN_HDF_Data
  d = EMAN_HDF_Data(path)

  glist = []
  for i in d.images:
    g = EMAN_HDF_Grid(d, i.size, i.value_type, i.origin, i.step, i.array_path)
    glist.append(g)

  return glist
