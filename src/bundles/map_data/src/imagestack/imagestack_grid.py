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
from .. import GridData
    
# -----------------------------------------------------------------------------
#
def image_stack_grids(paths):
  from . import imagestack_format
  d = imagestack_format.Image_Stack_Data(paths)
  if d.is_rgb:
    grids = [ImageStackGrid(d, channel) for channel in (0,1,2)]
    colors = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
    for g,rgba in zip(grids,colors):
      g.rgba = rgba
  else:
    grids = [ImageStackGrid(d)]
  return grids

# -----------------------------------------------------------------------------
#
class ImageStackGrid(GridData):

  def __init__(self, d, channel = None):

    self.image_stack = d

    GridData.__init__(self, d.data_size, d.value_type,
                      d.data_origin, d.data_step,
                      path = d.paths, file_type = 'imagestack', channel = channel)

  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):

    s = self.image_stack
    m = s.read_matrix(ijk_origin, ijk_size, ijk_step, self.channel, progress)
    return m
