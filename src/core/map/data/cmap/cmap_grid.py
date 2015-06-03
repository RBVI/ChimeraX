# -----------------------------------------------------------------------------
# Wrap Chimera HDF image data as grid data for displaying surface, meshes,
# and volumes.
#
from ..griddata import Grid_Data

# -----------------------------------------------------------------------------
#
class Chimera_HDF_Grid(Grid_Data):

  def __init__(self, hdf_data, size, value_type,
               origin, step, cell_angles, rotation, symmetries,
               image_name, array_paths):

    self.hdf_data = hdf_data
    self.array_paths = array_paths

    from os.path import basename
    name = basename(hdf_data.path)
    if image_name and image_name != name.rsplit('.',1)[0]:
      name += ' ' + image_name

    Grid_Data.__init__(self, size, value_type,
                       origin, step, cell_angles, rotation,
                       name = name, path = hdf_data.path, file_type = 'cmap',
                       grid_id = sorted(array_paths)[0])

    if not symmetries is None and len(symmetries) > 0:
      self.symmetries = symmetries

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import allocate_array
    m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    self.hdf_data.read_matrix(ijk_origin, ijk_size, ijk_step,
                              self.array_paths, m, progress)
    return m
    
# -----------------------------------------------------------------------------
#
def read_chimera_map(path):

  from .cmap_format import Chimera_HDF_Data
  d = Chimera_HDF_Data(path)

  glist = []
  for i in d.images:
    if len(d.images) > 1: image_name = i.name
    else:                 image_name = ''
    g = Chimera_HDF_Grid(d, i.size, i.value_type,
                         i.origin, i.step, i.cell_angles, i.rotation,
                         i.symmetries, image_name, i.array_paths)
    if i.subsamples:
      g = add_subsamples(d, i, g)
    glist.append(g)

  # Mark as volume series
  if len(glist) > 1 and len(set(tuple(g.size) for g in glist)) == 1:
      for i,g in enumerate(glist):
        g.series_index = i

  return glist
    
# -----------------------------------------------------------------------------
# Add subsample grids.
#
def add_subsamples(hdf_data, hdf_image, g):

  from ..subsample import Subsampled_Grid
  g = Subsampled_Grid(g)
  i = hdf_image
  for cell_size, data_size, array_paths in i.subsamples:
      step = tuple(map(lambda s,c: s*c, i.step, cell_size))
      sg = Chimera_HDF_Grid(hdf_data, data_size, i.value_type,
                            i.origin, step, i.cell_angles, i.rotation,
                            i.symmetries, i.name, array_paths)
      g.add_subsamples(sg, cell_size)
      
  return g
