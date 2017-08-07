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
from ..griddata import Grid_Data

# -----------------------------------------------------------------------------
#
class IMS_Grid(Grid_Data):

  def __init__(self, hdf_data, size, value_type,
               origin, step, cell_angles, rotation, symmetries, color,
               image_name, array_path):

    self.hdf_data = hdf_data
    self.array_path = array_path

    from os.path import basename
    name = basename(hdf_data.path)
    if image_name and image_name != name.rsplit('.',1)[0]:
      name += ' ' + image_name

    Grid_Data.__init__(self, size, value_type,
                       origin, step, cell_angles, rotation,
                       name = name, default_color = color,
                       path = hdf_data.path, file_type = 'ims', grid_id = array_path)

    if not symmetries is None and len(symmetries) > 0:
      self.symmetries = symmetries

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
    Grid_Data.clear_cache(self)
    
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
      g = IMS_Grid(d, i.size, i.value_type,
                   i.origin, i.step, i.cell_angles, i.rotation,
                   i.symmetries, i.default_color, image_name, i.array_path)
      if i.subsamples:
        g = add_subsamples(d, i, g)
      # Mark as volume series if maps of same size.
      glist.append(g)
    if len(glist) > 1 and len(set(tuple(g.size) for g in glist)) == 1:
      for i,g in enumerate(glist):
        g.series_index = i
    cglist.append(glist)


  return cglist
    
# -----------------------------------------------------------------------------
# Add subsample grids.
#
def add_subsamples(hdf_data, hdf_image, g):

  from ..subsample import Subsampled_Grid
  g = Subsampled_Grid(g)
  i = hdf_image
  for cell_size, data_size, array_path in i.subsamples:
      step = tuple(s*c for s,c in zip(i.step, cell_size))
      sg = IMS_Grid(hdf_data, data_size, i.value_type,
                    i.origin, step, i.cell_angles, i.rotation,
                    i.symmetries, i.default_color, i.name, array_path)
      g.add_subsamples(sg, cell_size)
      
  return g
