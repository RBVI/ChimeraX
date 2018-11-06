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
# Wrap Imaris image data as grid data for displaying surface, meshes, and volumes.
#
from ..griddata import GridData

# -----------------------------------------------------------------------------
#
class IMSGrid(GridData):

  # Attribute names for constructor grid settings.
  attributes = ('size', 'value_type', 'origin', 'step', 'cell_angles',
                'rotation', 'symmetries', 'default_color', 'time', 'channel')

  def __init__(self, hdf_data, image_name, array_path, **grid_settings):

    self.hdf_data = hdf_data
    self.array_path = array_path

    from os.path import basename
    name = basename(hdf_data.path)
    if image_name and image_name != name.rsplit('.',1)[0]:
      name += ' ' + image_name

    GridData.__init__(self, path = hdf_data.path, file_type = 'ims', grid_id = array_path, name = name,
                      **grid_settings)

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import allocate_array
    m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    self.hdf_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                              self.array_path, m, progress)
    return m

  # ---------------------------------------------------------------------------
  #
  def clear_cache(self):
    self.hdf_data.close_file()
    GridData.clear_cache(self)
    
# -----------------------------------------------------------------------------
#
def read_imaris_map(path):

  from .ims_format import IMS_Data
  d = IMS_Data(path)

  cglist = []
  for c,images in d.channel_images.items():
    glist = []
    for i in images:
      image_name = i.name if len(images) > 1 else ''
      gsettings = {attr:getattr(i,attr) for attr in IMSGrid.attributes}
      g = IMSGrid(d, image_name, i.array_path, **gsettings)
      if i.subsamples:
        g = add_subsamples(d, i, g)
      # Mark as volume series if maps of same size.
      glist.append(g)
    if len(glist) > 1 and len(set(tuple(g.size) for g in glist)) == 1:
      for i,g in enumerate(glist):
        g.series_index = i
    cglist.extend(glist)

  return cglist
    
# -----------------------------------------------------------------------------
# Add subsample grids.
#
def add_subsamples(hdf_data, hdf_image, g):

  from ..subsample import SubsampledGrid
  g = SubsampledGrid(g)
  i = hdf_image
  for cell_size, data_size, array_path in i.subsamples:
      step = tuple(s*c for s,c in zip(i.step, cell_size))
      gsettings = {attr:getattr(i,attr) for attr in IMSGrid.attributes}
      gsettings.update({'size':data_size, 'step':step})
      sg = IMSGrid(hdf_data, i.name, array_path, **gsettings)
      g.add_subsamples(sg, cell_size)
      
  return g
