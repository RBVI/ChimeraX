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
# Wrap Chimera HDF image data as grid data for displaying surface, meshes,
# and volumes.
#
from ..griddata import GridData

# -----------------------------------------------------------------------------
#
class ChimeraHDFGrid(GridData):

  # Attribute names for grid settings in constructor call.
  attributes = ('size', 'value_type', 'origin', 'step', 'cell_angles',
                'rotation', 'symmetries', 'default_color', 'time', 'channel')
  
  def __init__(self, hdf_data, image_name, array_paths, **grid_settings):

    self.hdf_data = hdf_data
    self.array_paths = array_paths

    from os.path import basename
    name = basename(hdf_data.path)
    if image_name and image_name != name.rsplit('.',1)[0]:
      name = image_name

    GridData.__init__(self, name = name, path = hdf_data.path, file_type = 'cmap',
                      grid_id = sorted(array_paths)[0], **grid_settings)

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import allocate_array
    m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    self.hdf_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                              self.array_paths, m, progress)
    return m

  # ---------------------------------------------------------------------------
  #
  def find_attribute(self, attribute_name):
    '''
    This is for finding segmentation attributes.
    If found in the HDF5 file it returns the attribute value (array, scalar, string).
    Returns None if the attribute is not found in the HDF5 file.
    If multiple attributes with the specified name are found it raises LookupError.
    '''
    return self.hdf_data.find_attribute(attribute_name)
    
# -----------------------------------------------------------------------------
#
def read_chimera_map(path):

  from .cmap_format import Chimera_HDF_Data
  d = Chimera_HDF_Data(path)

  glist = []
  for i in d.images:
    if len(d.images) > 1: image_name = i.name
    else:                 image_name = ''
    settings = {attr:getattr(i,attr) for attr in ChimeraHDFGrid.attributes}
    g = ChimeraHDFGrid(d, image_name, i.array_paths, **settings)
    if i.subsamples:
      g = add_subsamples(d, i, g)
    glist.append(g)

  _index_grid_series(glist)

  return glist

# -----------------------------------------------------------------------------
#
def _index_grid_series(grids):
  '''
  Mark as volume series if 5 or more maps of same size with the same channel number.
  Fewer than 5 maps are considered different maps.
  '''
  # Find grids by channel
  cgrids = {}
  for g in grids:
    if g.channel in cgrids:
      cgrids[g.channel].append(g)
    else:
      cgrids[g.channel] = [g]

  # Check each channel to see if it is a series.
  for glist in cgrids.values():
    if len(glist) > 4 and len(set(tuple(g.size) for g in glist)) == 1:
      for i,g in enumerate(glist):
        g.series_index = i
    
# -----------------------------------------------------------------------------
# Add subsample grids.
#
def add_subsamples(hdf_data, hdf_image, grid):

  from ..subsample import SubsampledGrid
  g = SubsampledGrid(grid)
  g.find_attribute = grid.find_attribute
  i = hdf_image
  for cell_size, data_size, array_paths in i.subsamples:
      settings = {attr:getattr(i,attr) for attr in ChimeraHDFGrid.attributes}
      step = tuple(s*c for s,c in zip(i.step, cell_size))
      settings.update({'size':data_size, 'step':step})
      sg = ChimeraHDFGrid(hdf_data, i.name, array_paths, **settings)
      g.add_subsamples(sg, cell_size)
      
  return g
