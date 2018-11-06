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
# Wrap netCDF array data as grid data for displaying surface, meshes,
# and volumes.
#
from .. import GridData
    
# -----------------------------------------------------------------------------
#
class NetCDFGrid(GridData):

  def __init__(self, path, grid_size, origin, step, cell_angles, rotation,
               netcdf_array):

    self.netcdf_array = netcdf_array

    from os.path import basename
    if netcdf_array.descriptive_name in ('', '0'):
      name = basename(path)
    else:
      name = '%s %s' % (basename(path), netcdf_array.descriptive_name)

    GridData.__init__(self, grid_size, netcdf_array.dtype,
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

  from .netcdf_format import NetCDF_Data
  d = NetCDF_Data(path)

  glist = []
  for a in d.arrays:
    g = NetCDFGrid(d.path, d.grid_size, d.xyz_origin, d.xyz_step,
                   d.cell_angles, d.rotation, a)
    ssa = subsampled_arrays(a)
    if ssa:
      from .. import SubsampledGrid
      g = SubsampledGrid(g)
      for grid_size, cell_size, arrays in ssa:
        step = [s*c for s,c in zip(d.xyz_step, cell_size)]
        sg = NetCDFGrid(d.path, grid_size, d.xyz_origin, step,
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
