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
class DicomData:

  def __init__(self, paths, verbose = False):

    if isinstance(paths, str):
      paths = indexed_files(paths)
      if not paths:
        raise SyntaxError('No files found %s' % path)
    self.paths = tuple(ordered_paths(paths))

    import pydicom
    d = pydicom.dcmread(self.paths[0])

#    for attr in ['AccessionNumber', 'BitsAllocated', 'BitsStored', 'Columns', 'HighBit', 'ImageOrientationPatient', 'ImagePositionPatient', 'ImageType', 'InstanceCreationDate', 'InstanceCreationTime', 'InstanceNumber', 'Modality', 'PatientBirthDate', 'PatientID', 'PatientName', 'PatientSex', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing', 'ReferringPhysicianName', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SeriesDate', 'SeriesDescription', 'SeriesInstanceUID', 'SeriesNumber', 'SeriesTime', 'StudyDate', 'StudyID', 'StudyInstanceUID', 'StudyTime']:
#      print (attr, getattr(d, attr, 'no value'))
    if verbose:
      print(d)

#      print ('spacing', d.SpacingBetweenSlices)
#    print('dicom file dir', dir(d))

    rsi = float(getattr(d, 'RescaleIntercept', 0))
    if rsi == int(rsi):
      rsi = int(rsi)
    self.rescale_intercept = rsi
    self.rescale_slope = float(getattr(d, 'RescaleSlope', 1))

    self.value_type = numpy_value_type(d.BitsAllocated, d.PixelRepresentation,
                                       self.rescale_slope, self.rescale_intercept)
    if d.SamplesPerPixel == 1:
      mode = 'grayscale'
    elif d.SamplesPerPixel == 3:
      mode = 'RGB'
    else:
      raise ValueError('Only 1 or 3 samples per pixel supported, got', d.SamplesPerPixel)
    self.mode = mode
    self.channel = 0
    if d.PhotometricInterpretation == 'MONOCHROME1':
      pass # Bright to dark values.
    if d.PhotometricInterpretation == 'MONOCHROME2':
      pass # Dark to bright values.

    if hasattr(d, 'PixelPaddingValue'):
      self.pad_value = self.rescale_slope * d.PixelPaddingValue + self.rescale_intercept
      print('padding value', self.pad_value)
    else:
      self.pad_value = None

    self.reverse_stack = False
    n = int(d.InstanceNumber)
    x,y,z = [float(p) for p in d.ImagePositionPatient]
    zs = 1
    if len(self.paths) > 1:
      # TODO: Need to read every image because file numbering may
      # not match image number and image position.  Also need to
      # check that images are uniformly spaced.
      d2 = pydicom.dcmread(self.paths[1])
      n2 = int(d2.InstanceNumber)
      x2,y2,z2 = [float(p) for p in d2.ImagePositionPatient]
      if n2 == n + 1 or n2 == n - 1:
        zs = abs(z2 - z)
        if z2 < z:
          self.reverse_stack = True
      else:
        print('Did not get z spacing, first two images were not consecutive', n, n2)

    xsize, ysize = d.Columns, d.Rows

    self.data_size = (xsize, ysize, len(self.paths))
    xs, ys = [float(s) for s in d.PixelSpacing]
    self.data_step = (xs, ys, zs)
    self.data_origin = (0.0, 0.0, 0.0)

  # ---------------------------------------------------------------------------
  # Reads a submatrix and returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, channel, array, progress):

    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    dsize = self.data_size
    for k in range(k0, k0+ksz, kstep):
      if progress:
        progress.plane((k-k0)//kstep)
      p = self.read_plane(k, channel)
      array[(k-k0)//kstep,:,:] = p[j0:j0+jsz:jstep,i0:i0+isz:istep]

    if self.rescale_slope != 1:
      array *= self.rescale_slope
    if self.rescale_intercept != 0:
      array += self.rescale_intercept
    return array

  # ---------------------------------------------------------------------------
  #
  def read_plane(self, k, channel = None):
    p = self.data_size[2]-1-k if self.reverse_stack else k
    import pydicom
    d = pydicom.dcmread(self.paths[p])
    data = d.pixel_array
    if channel is not None:
      data = data[:,:,channel]
    return data

# -----------------------------------------------------------------------------
# Find files matching wildcard pattern.
#
def indexed_files(path):

  if '*' in path or '?' in path:
    import glob
    paths = glob.glob(path)
    return paths

  return [path]

# -----------------------------------------------------------------------------
#
def ordered_paths(paths):
  opaths = list(paths)
  prefix, suffix = common_prefix_and_suffix(opaths)
#  print('common prefix, suffix', prefix, suffix)
  pi = len(prefix)
  si = len(suffix)
  if integer_sequence([p[pi:-si] for p in opaths]):
#    print ('int seq')
    # Sort by integer in filename that may not be zero padded.
    opaths.sort(key = lambda p: int(p[pi:-si]))
  else:
    # Sort alphabetically
    opaths.sort()
  return opaths

# -----------------------------------------------------------------------------
#
def common_prefix_and_suffix(strings):
  prefix = suffix = strings[0]
  for s in strings[1:]:
    if not s.startswith(prefix):
      for i in range(min(len(prefix), len(s))):
        if s[i] != prefix[i]:
          prefix = prefix[:i]
          break
    if not s.endswith(suffix):
      for i in range(min(len(suffix), len(s))):
        if s[-1-i] != suffix[-1-i]:
          suffix = suffix[len(suffix)-i:]
          break
  return prefix, suffix

# -----------------------------------------------------------------------------
# Are all strings integers.
#
def integer_sequence(strings):
  try:
    for s in strings:
      int(s)
  except ValueError:
    return False
  return True

# -----------------------------------------------------------------------------
# PixelRepresentation 0 = unsigned, 1 = signed
#
def numpy_value_type(bits_allocated, pixel_representation, rescale_slope, rescale_intercept):

  from numpy import int8, uint8, int16, uint16, float32
  if rescale_slope != 1 or int(rescale_intercept) != rescale_intercept:
    return float32

  types = {(8,0): uint8,
           (8,1): int8,
           (16,0): uint16,
           (16,1): int16}
  if (bits_allocated, pixel_representation) in types:
    return types[(bits_allocated, pixel_representation)]

  raise ValueError('Unsupported value type, bits_allocated = ', bits_allocated)
