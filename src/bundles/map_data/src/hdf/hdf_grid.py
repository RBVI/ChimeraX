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
# Wrap HDF image data as grid data for displaying surface, meshes, and volumes.
#
from ..griddata import GridData

# -----------------------------------------------------------------------------
#
class HDFGrid(GridData):

  def __init__(self, hdf_data, image_name, array_path, size, value_type):

    self.hdf_data = hdf_data
    self.array_path = array_path

    from os.path import basename
    name = basename(hdf_data.path)
    if image_name and image_name != name.rsplit('.',1)[0]:
      name = image_name

    GridData.__init__(self, size, value_type = value_type,
                      name = name, path = hdf_data.path, file_type = 'hdf',
                      grid_id = array_path)

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
def read_hdf_map(path, array_name = None):

  from .hdf_format import HDFData
  d = HDFData(path)

  images = d.images
  if len(images) == 0:
    raise SyntaxError('HDF map reader: No 3D arrays found in file %s' % path)
  if array_name is not None:
    images = [i for i in images if i.name == array_name]
    if len(images) == 0:
      raise SyntaxError('HDF map reader: No 3D arrays with name "%s"\nfound in file %s,\nfound names %s'
                        % (array_name, path, ', '.join(i.name for i in d.images)))

  glist = []
  for i in images:
    image_name = i.name if len(images) > 1 else ''
    g = HDFGrid(d, image_name, i.array_path, size = i.size, value_type = i.value_type)
    glist.append(g)

  # Mark as volume series if 5 or more maps of same size.
  # Fewer than 5 maps are considered different channels.
  if len(glist) > 4 and len(set(tuple(g.size) for g in glist)) == 1:
      for i,g in enumerate(glist):
        g.series_index = i

  return glist
