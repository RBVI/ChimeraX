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
