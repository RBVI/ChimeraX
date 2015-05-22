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

    from PIL import Image
    i = Image.open(self.paths[0])

    if i.mode in modes:
      self.value_type = modes[i.mode]
      self.mode = i.mode
    else:
      mnames = ', '.join(modes.keys())
      raise SyntaxError('Image mode %s is not supported (%s)' % (i.mode, mnames))
    self.channel = 0

    xsize, ysize = i.size

    if len(self.paths) == 1:
      icount = image_count(i)
      self.is_multipage = (icount > 1)
    else:
      icount = len(self.paths)
      self.is_multipage = False

    self.data_size = (xsize, ysize, icount)
    self.data_step = (1.0, 1.0, 1.0)
    self.data_origin = (0.0, 0.0, 0.0)

  # ---------------------------------------------------------------------------
  # Reads a submatrix and returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, channel, array, progress):

    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    dsize = self.data_size
    mim = self.multipage_image()
    for k in range(k0, k0+ksz, kstep):
      if progress:
        progress.plane((k-k0)/kstep)
      p = self.read_plane(k, mim, channel)
      array[(k-k0)/kstep,:,:] = p[j0:j0+jsz:jstep,i0:i0+isz:istep]
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

  from PIL import Image
  i = Image.open(path)
  return image_count(i, max = 2) > 1

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
