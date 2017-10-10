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
      tiff_type = tiff_format(p)
      if tiff_type == 'OME':
        from . import ome_tiff
        grids.extend(ome_tiff.ome_image_grids(p))
        series_check = False
      elif tiff_type == 'ImageJ':
        from . import imagej_tiff
        grids.extend(imagej_tiff.imagej_grids(p))
      else:
        grids.extend(image_stack_grids([p]))

    assign_series_and_channels(grids)

    # Assign default channel colors
    for g in grids:
      if g.rgba is None and g.channel is not None:
        g.rgba = default_channel_colors[g.channel % len(default_channel_colors)]

  return grids

# -----------------------------------------------------------------------------
#
def assign_series_and_channels(grids):
  assigned = False
  for g in grids:
    channel = g.channel
    series_index = getattr(g, 'series_index', g.time)
    if channel is not None and series_index is not None:
      continue
    gpath = g.path if isinstance(g.path, str) else g.path[0]
    fields = gpath.split('_')
    for f in fields:
      if channel is None and f.startswith('ch') and is_integer(f[2:]):
        channel = int(f[2:])
        assigned = True
      elif channel is None and f.endswith('ch') and is_integer(f[:-2]):
        channel = int(f[:-2])
        assigned = True
      elif series_index is None and f.startswith('stack') and is_integer(f[5:]):
        series_index = int(f[5:])
        assigned = True
      elif series_index is None and f.endswith('stack') and is_integer(f[:-5]):
        series_index = int(f[:-5])
        assigned = True
    if channel is not None:
      g.channel = channel
    if series_index is not None:
      g.series_index = series_index

  # Mark volume series if all grids the same size
  if not assigned:
    channels = set(g.channel for g in grids)
    for c in channels:
      gc = tuple(g for g in grids if g.channel == channel)
      if len(gc) > 1 and len(set(tuple(g.size) for g in gc)) == 1:
        for i,g in enumerate(gc):
          g.series_index = i

# -----------------------------------------------------------------------------
#
def is_integer(string):
  try:
    int(string)
  except ValueError:
    return False
  return True

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

default_channel_colors = [
    (1,0,0,1),
    (0,1,0,1),
    (0,0,1,1),
    (1,1,0,1),
    (1,0,1,1),
    (0,1,1,1),
    (.5,0,0,1),
    (0,.5,0,1),
    (0,0,.5,1),
    (.5,1,0,1),
    (1,.5,0,1),
    (.5,0,1,1),
    (1,0,.5,1),
    (0,.5,1,1),
    (0,1,.5,1),
]
