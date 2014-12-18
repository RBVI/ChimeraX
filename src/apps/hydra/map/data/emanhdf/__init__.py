# -----------------------------------------------------------------------------
# EMAN HDF map file reader.
#
def open(path):

  from eman_hdf_grid import read_eman_hdf5
  return read_eman_hdf5(path)
