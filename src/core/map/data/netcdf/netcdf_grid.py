# -----------------------------------------------------------------------------
# Wrap netCDF array data as grid data for displaying surface, meshes,
# and volumes.
#
from .. import Grid_Data
    
# -----------------------------------------------------------------------------
#
class NetCDF_Grid(Grid_Data):

  def __init__(self, path, grid_size, origin, step, cell_angles, rotation,
               netcdf_array):

    self.netcdf_array = netcdf_array

    from os.path import basename
    if netcdf_array.descriptive_name in ('', '0'):
      name = basename(path)
    else:
      name = '%s %s' % (basename(path), netcdf_array.descriptive_name)

    Grid_Data.__init__(self, grid_size, netcdf_array.dtype,
                       origin, step, cell_angles, rotation,
                       name = name, path = path, file_type = 'netcdf', 
                       grid_id = netcdf_array.variable_name,
                       default_color = netcdf_array.color)
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    m = self.netcdf_array.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
    return m
    
# -----------------------------------------------------------------------------
#
def read_netcdf(path):

  from netcdf_format import NetCDF_Data
  d = NetCDF_Data(path)

  glist = []
  for a in d.arrays:
    g = NetCDF_Grid(d.path, d.grid_size, d.xyz_origin, d.xyz_step,
                    d.cell_angles, d.rotation, a)
    ssa = subsampled_arrays(a)
    if ssa:
      from . import Subsampled_Grid
      g = Subsampled_Grid(g)
      for grid_size, cell_size, arrays in ssa:
        step = map(lambda s,c: s*c, d.xyz_step, cell_size)
        sg = NetCDF_Grid(d.path, grid_size, d.xyz_origin, step,
                         d.cell_angles, d.rotation, arrays)
        g.add_subsamples(sg, cell_size)
    glist.append(g)
    
  return glist
    
# -----------------------------------------------------------------------------
#
def subsampled_arrays(netcdf_array):

  a = netcdf_array
  if not hasattr(a, 'subsamples'):
    return []

  ssa = [(gc_size[0], gc_size[1], alist)
         for gc_size, alist in a.subsamples.items()]
  return ssa
