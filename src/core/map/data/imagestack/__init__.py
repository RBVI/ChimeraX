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
    series_check = True
    for p in paths:
      tiff_type = tiff_format(p) if p.endswith('.tif') else None
      if tiff_type == 'OME':
        from . import ome_tiff
        grids.extend(ome_tiff.ome_image_grids(p))
        series_check = False
      elif tiff_type == 'ImageJ':
        from . import imagej_tiff
        grids.extend(imagej_tiff.imagej_grids(p))
      else:
        grids.extend(image_stack_grids([p]))

    # Mark volume series
    if series_check and len(grids) > 1 and len(set(tuple(g.size) for g in grids)) == 1:
      for i,g in enumerate(grids):
        g.series_index = i

  return grids

# -----------------------------------------------------------------------------
# Look for OME and ImageJ Tiff files which have extra header info and can
# contain multiple channels.
#
def tiff_format(path):
  if path.endswith('.ome.tif') or path.endswith('.ome.tiff'):
    return 'OME'
  if path.endswith('.tif') or path.endswith('.tiff'):
    from PIL import Image
    i = Image.open(path)
    description_tags = i.tag[270]
    for d in description_tags:
      if d.startswith('<?xml'):
        return 'OME'
      elif d.startswith('ImageJ='):
        return 'ImageJ'
  return None
