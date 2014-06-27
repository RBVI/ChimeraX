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
        raise SyntaxError, 'No files found %s' % path
    self.paths = tuple(paths)

    from numpy import uint8, uint16, int32, float32, little_endian
    modes = {'L': uint8,
             'P': uint8,
             'I;16': uint16,
             'I;16B': uint16,   # Big endian, converted to native by PIL
             'I;16L': uint16,   # Little endian, converted to native by PIL
             'F': float32,
             'I': int32,        # TODO: Don't have any test data for I.
             }

    from PIL import Image
    i = Image.open(self.paths[0])

    if i.mode in modes:
      self.value_type = modes[i.mode]
    else:
      mnames = ', '.join(modes.keys())
      raise SyntaxError, ('Image mode %s is not supported (%s)'
                          % (i.mode, mnames))
    xsize, ysize = i.size

    self.multipage_image = None
    if len(self.paths) == 1:
      icount = image_count(i)
      if icount > 1:
        self.multipage_image = i
    else:
      icount = len(self.paths)

    self.data_size = (xsize, ysize, icount)
    self.data_step = (1.0, 1.0, 1.0)
    self.data_origin = (0.0, 0.0, 0.0)

  # ---------------------------------------------------------------------------
  # Reads a submatrix and returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, array, progress):

    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    dsize = self.data_size
    from numpy import zeros
    ia = zeros((dsize[1],dsize[0]), self.value_type)
    ia_1d = ia.ravel()
    for k in range(k0, k0+ksz, kstep):
      if progress:
        progress.plane((k-k0)/kstep)
      ia_1d[:] = self.read_plane(k)
      array[(k-k0)/kstep,:,:] = ia[j0:j0+jsz:jstep,i0:i0+isz:istep]
    return array

  # ---------------------------------------------------------------------------
  #
  def read_plane(self, k):

    if self.multipage_image is None:
      from PIL import Image
      im = Image.open(self.paths[k])
    else:
      im = self.multipage_image
      im.seek(k)
    return im.getdata()

# -----------------------------------------------------------------------------
# Count images in a possibly multi-page PIL image.
#
def image_count(image):

  frame = 0
  try:
    while True:
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
