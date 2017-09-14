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
# Wrap image data as grid data for displaying surface, meshes, and volumes.
#
from .. import Grid_Data
    
# -----------------------------------------------------------------------------
#
def image_stack_grids(paths):
  from . import imagestack_format
  d = imagestack_format.Image_Stack_Data(paths)
  if d.mode == 'RGB':
    grids = [Image_Stack_Grid(d, channel) for channel in (0,1,2)]
    colors = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
    for g,rgba in zip(grids,colors):
      g.rgba = rgba
  else:
    grids = [Image_Stack_Grid(d)]
  return grids

# -----------------------------------------------------------------------------
#
class Image_Stack_Grid(Grid_Data):

  def __init__(self, d, channel = 0):

    self.image_stack = d

    Grid_Data.__init__(self, d.data_size, d.value_type,
                       d.data_origin, d.data_step,
                       path = d.paths, file_type = 'imagestack', channel = channel)

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    from ..readarray import allocate_array
    m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    self.image_stack.read_matrix(ijk_origin, ijk_size, ijk_step, self.channel, m, progress)
    return m
