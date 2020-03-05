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
# Wrap EMAN HDF image data as grid data for displaying surface, meshes,
#  and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class EMANHDFGrid(GridData):

  def __init__(self, eman_hdf_data, size, value_type,
               origin, step, array_path):

    self.eman_hdf_data = eman_hdf_data
    self.array_path = array_path

    GridData.__init__(self, size, value_type,
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
def read_eman_hdf5(path, array_name = None):

  from .eman_hdf_format import EMAN_HDF_Data
  d = EMAN_HDF_Data(path)

  if len(d.images) == 0:
    # Didn't find any EMAN HDF5 3d arrays.  Try generic HDF reader.
    from .. import hdf
    return hdf.open(path, array_name = array_name)

  glist = []
  for i in d.images:
    g = EMANHDFGrid(d, i.size, i.value_type, i.origin, i.step, i.array_path)
    glist.append(g)

  return glist
