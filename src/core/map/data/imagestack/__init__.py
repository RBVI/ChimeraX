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
# Image stack file reader.
#
def open(paths):

  from .imagestack_grid import image_stack_grids
  from .imagestack_format import is_3d_image
  if isinstance(paths, str):
    paths = [paths]

  if not is_3d_image(paths[0]):
    # Stack of 2d images as a single map.
    grids = image_stack_grids(paths)
  else:
    # 3d images.
    grids = []
    ome = False
    for p in paths:
      if p.endswith('.ome.tif') or p.endswith('.ome.tiff'):
        from . import ome_tiff
        grids.extend(ome_tiff.ome_image_grids(p))
        ome = True
      else:
        grids.extend(image_stack_grids([p]))

    # Mark volume series
    if not ome and len(grids) > 1 and len(set(tuple(g.size) for g in grids)) == 1:
      for i,g in enumerate(grids):
        g.series_index = i

  return grids
