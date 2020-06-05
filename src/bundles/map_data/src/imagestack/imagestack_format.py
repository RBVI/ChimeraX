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
# Read stack of images representing a density map.  The image filename
# suffix gives the image number.  Images are stacked increasing order of
# image number.  Gaps in numbering are allowed but do not create a gap in
# the volume.
#

# -----------------------------------------------------------------------------
#
class Image_Stack_Data:

  def __init__(self, paths):

    if isinstance(paths, str):
      paths = indexed_files(paths)
      if not paths:
        raise SyntaxError('No files found %s' % path)
    self.paths = tuple(paths)

    self.is_tiff = len([p for p in self.paths if p.endswith('.tif') or p.endswith('.tiff')]) == len(self.paths)

    if self.is_tiff:
      # For TIFF images use tifffile.py
      from tifffile import TiffFile, TIFF
      tif = TiffFile(self.paths[0])
      pages = tif.pages
      page0 = pages[0]
      value_type = page0.dtype
      is_rgb = (page0.photometric == TIFF.PHOTOMETRIC.RGB)

      ysize, xsize = page0.shape[:2]

      if len(self.paths) == 1:
        zsize = len(pages)
        is_multipage = (zsize > 1)
      else:
        zsize = len(self.paths)
        is_multipage = False

    else:
      # For images other than TIFF use Pillow.
      from PIL import Image
      i = Image.open(self.paths[0])

      value_type = pillow_numpy_value_type(i.mode)
      is_rgb = (i.mode == 'RGB')

      xsize, ysize = i.size

      if len(self.paths) == 1:
        zsize = image_count(i)
        is_multipage = (zsize > 1)
      else:
        zsize = len(self.paths)
        is_multipage = False

    self.value_type = value_type
    self.is_rgb = is_rgb
    self.is_multipage = is_multipage
    self.data_size = (xsize, ysize, zsize)
    self.data_step = (1.0, 1.0, 1.0)
    self.data_origin = (0.0, 0.0, 0.0)

  # ---------------------------------------------------------------------------
  # Reads a submatrix and returns 3D NumPy matrix with zyx index order.
  #
  def read_tiff_matrix(self, ijk_origin, ijk_size, ijk_step, channel, progress):

    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    klist = range(k0, k0+ksz, kstep)
    if self.is_multipage:
      from tifffile import TiffFile
      with TiffFile(self.paths[0]) as tif:
        a = tif.asarray(key = klist)
    else:
      from tifffile import imread
      a = imread([self.paths[k] for k in klist])
    if self.is_rgb and channel is not None:
      if a.ndim == 4:
        a = a[:,:,:,channel]
      elif a.ndim == 3:
        a = a[:,:,channel]
    if a.ndim == 2:
      a = a.reshape((1,) + tuple(a.shape))	# Make single-plane 3d
    array = a[:, j0:j0+jsz:jstep,i0:i0+isz:istep]
    return array

  # ---------------------------------------------------------------------------
  # Reads a submatrix and returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, channel, progress):
    if self.is_tiff:
      return self.read_tiff_matrix(ijk_origin, ijk_size, ijk_step, channel, progress)

    # Read using Pillow.
    from ..readarray import allocate_array
    array = allocate_array(ijk_size, self.value_type, ijk_step, progress)
    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    mim = self.multipage_image()
    for k in range(k0, k0+ksz, kstep):
      if progress:
        progress.plane((k-k0)//kstep)
      p = self.read_plane(k, mim, channel)
      array[(k-k0)//kstep,:,:] = p[j0:j0+jsz:jstep,i0:i0+isz:istep]
    return array

  # ---------------------------------------------------------------------------
  #
  def read_plane(self, k, multipage_image = None, channel = 0):

    im = multipage_image
    if im is None:
      from PIL import Image
      im = Image.open(self.paths[k])
    else:
      im.seek(k)
    from numpy import array
    a = array(im)
    if a.ndim == 3:
      return a[:,:,channel]
    return a

  # ---------------------------------------------------------------------------
  #
  def multipage_image(self):

    if self.is_multipage:
      from PIL import Image
      im = Image.open(self.paths[0])
    else:
      im = None
    return im

# -----------------------------------------------------------------------------
#
def is_3d_image(path):

  if path.endswith('.tif') or path.endswith('.tiff'):
    from tifffile import TiffFile
    with TiffFile(path) as tif:
      is_3d = True
      try:
        tif.pages[1]
      except IndexError:
        is_3d = False
  else:
    # Handle non-TIFF images using Pillow.
    from PIL import Image
    i = Image.open(path)
    is_3d = image_count(i, max = 2) > 1
  return is_3d

# -----------------------------------------------------------------------------
# Count images in a possibly multi-page PIL image.
#
def image_count(image, max = None):

  frame = 0
  try:
    while max is None or frame < max:
      image.seek(frame)
      frame += 1
  except EOFError:
    image.seek(0)
  except ValueError as e:
    # PIL 7.1.1 raised this error on large single plane image.
    if 'read of closed file' not in str(e):
      raise
  return frame

# -----------------------------------------------------------------------------
# Find files matching wildcard pattern.
#
def indexed_files(path):

  if '*' in path or '?' in path:
    import glob
    paths = glob.glob(path)
    paths.sort()
    return paths

  return [path]

# -----------------------------------------------------------------------------
#
def pillow_numpy_value_type(image_mode):

    from numpy import uint8, uint16, int16, int32, uint32, float32, little_endian
    modes = {'L': uint8,
             'P': uint8,
             'I;16': uint16,
             'I;16B': uint16,   # Big endian, converted to native by PIL
             'I;16L': uint16,   # Little endian, converted to native by PIL
             'I;16S': int16,    # Signed 16-bit
             'F': float32,
             'F;32BF': float32,	# Big endian, not sure if PIL converts this.
             'I': int32,        # TODO: Don't have any test data for I.
             'RGB': uint8,
             }
    if image_mode not in modes:
      mnames = ', '.join(modes.keys())
      raise SyntaxError('Image mode %s is not supported (%s)' % (image_mode, mnames))

    return modes[image_mode]
    
