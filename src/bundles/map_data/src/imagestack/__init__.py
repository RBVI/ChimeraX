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
def open(paths, log=None):

  set_maximum_pil_image_size(None)	# No Python Image Library image size limit.
  
  if isinstance(paths, str):
    paths = [paths]

  tc_paths = paths_by_time_and_channel(paths)
  grids = []
  for t,c,path_list in tc_paths:
    grids.extend(open_stack(path_list, t, c, log=log))

  # Assign default channel colors
  gc = [g for g in grids if g.rgba is None and g.channel is not None]
  if len(set(g.channel for g in gc)) > 1:
    for g in gc:
      g.rgba = default_channel_colors[g.channel % len(default_channel_colors)]

  return grids

# -----------------------------------------------------------------------------
#
def open_stack(paths, time=None, channel=None, log=None):

  from .imagestack_grid import image_stack_grids
  from .imagestack_format import is_3d_image
  if len(paths) > 1 and not is_3d_image(paths[0]):
    # Stack of 2d images as a single map.
    grids = image_stack_grids(paths)
  else:
    # 3d images.
    grids = []
    series_check = True
    fpaths = set()
    for p in paths:
      tiff_type = tiff_format(p)
      if tiff_type == 'OME':
        if p in fpaths:
          continue	# OME file already referenced by a previous OME file.
        from . import ome_tiff
        pgrids = ome_tiff.ome_image_grids(p, fpaths, log=log)
        series_check = False
      elif tiff_type == 'ImageJ':
        from . import imagej_tiff
        pgrids = imagej_tiff.imagej_grids(p)
      else:
        pgrids = image_stack_grids([p])
      grids.extend(pgrids)

  if time is None and channel is None:
    assign_series_and_channels(grids)
  else:
    for g in grids:
      g.time = g.series_index = time
      g.channel = channel

  return grids

# -----------------------------------------------------------------------------
#
def paths_by_time_and_channel(paths):

  tc_paths = []
  reg_paths = []
  from os.path import split
  for path in paths:
    dir, filename = split(path)
    if '{t}' in filename or '{c}' in filename or '{z}' in filename:
      tc_zpaths = parse_path_tcz(dir, filename)
      if len(tc_zpaths) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No file matching pattern not found: %s' % path)
      tc_paths.extend(tc_zpaths)
    else:
      from glob import glob
      gpaths = glob(path)
      if len(gpaths) == 0:
        from chimerax.core.errors import UserError
        raise UserError('File not found: %s' % path)
      reg_paths.extend(gpaths)

  if reg_paths:
    tc_paths.append((None, None, reg_paths))

  return tc_paths

# -----------------------------------------------------------------------------
#
def parse_path_tcz(dir, filename):

  # Make a regular expression to match filenames.
  pattern = filename
  fields = []
  for c in ('t', 'c', 'z'):
    field = '{%s}' % c
    if field in pattern:
      pattern = pattern.replace(field,'(?P<%s>[0-9]+)' % c)
      fields.append(c)
  if '*' in pattern:
    pattern = pattern.replace('*', '.*?')	# Convert glob * to regex equivalent.
  import re
  rexpr = re.compile(pattern)

  from os import listdir
  filenames = listdir(dir if dir else '.')

  # Collect all paths for each time and channel and sort in z order.
  tc_zpaths = {}
  for fname in filenames:
    m = rexpr.fullmatch(fname)
    if m:
      tcz = {}
      for c in fields:
        value = m.group(c)
        tcz[c] = int(value)
      tc = (tcz.get('t'), tcz.get('c'))
      if tc not in tc_zpaths:
        tc_zpaths[tc] = []
      tc_zpaths[tc].append((tcz.get('z'), fname))

  from os.path import join
  tc_zpaths_list = [(t,c,[join(dir,fname) for z,fname in sorted(zpaths)])
                    for (t,c),zpaths in tc_zpaths.items()]
  tc_zpaths_list.sort(key = lambda tczs: tczs[:2])
                      
  return tc_zpaths_list
    
    
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

  # Mark volume series if all grids the same size.
  # If 4 or fewer grids with channel None then assign them as channels.
  if not assigned:
    channels = set(g.channel for g in grids)
    for c in channels:
      gc = tuple(g for g in grids if g.channel == c)
      if len(gc) > 1 and len(set(tuple(g.size) for g in gc)) == 1:
        if c is None and len(gc) <= 4:
          for i,g in enumerate(gc):
            g.channel = i
        else:
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
    from tifffile import TiffFile
    with TiffFile(path) as tif:
        tags = tif.pages[0].tags
        desc = tags['ImageDescription'].value if 'ImageDescription' in tags else None
    if desc is None:
      return None
    elif desc.startswith('<?xml'):
      return 'OME'
    elif desc.startswith('ImageJ='):
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

# -----------------------------------------------------------------------------
#
def set_maximum_pil_image_size(pixels):
  '''
  Maximum image pixels in Pillow 7.1.1 single image is 89 million = (2**30/12)
  which is an image size of only 9459 x 9459.  Pillow raises an exception 
  DecompressionBombWarning on larger images. I tried setting it to 2**32 and
  it opened a 1.2 billion pixel image (Bennu asteroid terrain).
  '''
  from PIL import Image
  Image.MAX_IMAGE_PIXELS = pixels


# -----------------------------------------------------------------------------
#
from .imagej_write import write_imagej_tiff as save

