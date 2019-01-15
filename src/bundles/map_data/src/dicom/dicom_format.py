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
# Look through directories to find dicom files (.dcm) and group the ones
# that belong to the same study and image series.  Also determine the order
# of the 2D images (one per file) in the 3D stack.  A series must be in a single
# directory.  If the same study and series is found in two directories, they
# are treated as two different series.
#
def find_dicom_series(paths, search_directories = True, search_subdirectories = True, verbose = False):
  dfiles = files_by_directory(paths, search_directories = search_directories,
                              search_subdirectories = search_subdirectories)
  series = []
  for dpaths in dfiles.values():
    series.extend(dicom_file_series(dpaths, verbose = verbose))
  return series

# -----------------------------------------------------------------------------
# Group dicom files into series.
#
def dicom_file_series(paths, verbose = False):
  series = {}
  from pydicom import dcmread
  for path in paths:
    d = dcmread(path)
    if hasattr(d, 'SeriesInstanceUID'):
      series_id = d.SeriesInstanceUID
      if series_id in series:
        s = series[series_id]
      else:
        series[series_id] = s = Series()
        if verbose:
          print ('Data set: %s\n%s\n' % (path, d))
      s.add(path, d)
  series = tuple(series.values())
  for s in series:
    s.order_slices()
  return series

# -----------------------------------------------------------------------------
# Set of dicom files (.dcm suffix) that have the same series unique identifer (UID).
#
class Series:
  dicom_attributes = ['BitsAllocated', 'Columns', 'Modality',
                      'PatientID', 'PhotometricInterpretation',
                      'PixelPaddingValue', 'PixelRepresentation', 'PixelSpacing',
                      'RescaleIntercept', 'RescaleSlope', 'Rows',
                      'SamplesPerPixel', 'SeriesDescription', 'SeriesInstanceUID',
                      'StudyDate']
  def __init__(self):
    self.paths = []
    self.positions = []
    self.nums = []
    self.attributes = {}
    self.num_times = 1
  def add(self, path, data):
    if len(self.paths) == 0:
      attrs = self.attributes
      for attr in self.dicom_attributes:
        if hasattr(data, attr):
          attrs[attr] = getattr(data, attr)

    self.paths.append(path)

    pos = getattr(data, 'ImagePositionPatient', None)
    xyz = tuple(float(p) for p in pos) if pos else None
    self.positions.append(xyz)
    
    num = getattr(data, 'InstanceNumber', None)
    n = int(num) if num else None
    self.nums.append(n)

    nt = getattr(data, 'NumberOfTemporalPositions', None)
    if nt is not None:
      self.num_times = int(nt)
      
  def order_slices(self):
    paths = self.paths
    if len(paths) <= 1:
      return paths
    
    for path,num in zip(paths, self.nums):
      if num is None:
        raise ValueError("Missing dicom InstanceNumber, can't order slice %s" % path)

    # Order slices by number, but flip if z values are decreasing.
    from numpy import argsort
    si = argsort(self.nums)
    z0, z1 = self.positions[si[0]][2], self.positions[si[1]][2]
    if z0 > z1:
      from numpy import flip
      si = flip(si, axis=0)
    self.paths = tuple(paths[i] for i in si)
    self.positions = tuple(self.positions[i] for i in si)
    self.nums = tuple(self.nums[i] for i in si)

  def z_plane_spacing(self):
    pos = self.positions
    if len(pos) < 2:
      dz = None
    else:
      dz = pos[1][2] - pos[0][2]
    return dz
  
# -----------------------------------------------------------------------------
# Find all dicom files (suffix .dcm) in directories and subdirectories and
# group them by directory.
#
def files_by_directory(paths, search_directories = True, search_subdirectories = True,
                       suffix = '.dcm', _dfiles = None):
  dfiles = {} if _dfiles is None else _dfiles
  from os.path import isfile, isdir, dirname, join
  from os import listdir
  for p in paths:
    if isfile(p) and p.endswith(suffix):
      d = dirname(p)
      if d in dfiles:
        dfiles[d].add(p)
      else:
        dfiles[d] = set([p])
    elif search_directories and isdir(p):
      ppaths = [join(p,fname) for fname in listdir(p)]
      files_by_directory(ppaths, search_directories=search_subdirectories,
                         search_subdirectories=search_subdirectories, _dfiles=dfiles)
  return dfiles
  
# -----------------------------------------------------------------------------
#
class DicomData:

  def __init__(self, series):

    self.paths = series.paths

    attrs = series.attributes
    name = '%s %s %s' % (attrs.get('PatientID', '')[:4],
                         attrs.get('SeriesDescription', ''),
                         attrs.get('StudyDate', ''))
    self.name = name

    
    rsi = float(attrs.get('RescaleIntercept', 0))
    if rsi == int(rsi):
      rsi = int(rsi)
    self.rescale_intercept = rsi
    self.rescale_slope = float(attrs.get('RescaleSlope', 1))

    self.value_type = numpy_value_type(attrs['BitsAllocated'], attrs['PixelRepresentation'],
                                       self.rescale_slope, self.rescale_intercept)
    ns = attrs['SamplesPerPixel']
    if ns == 1:
      mode = 'grayscale'
    elif ns == 3:
      mode = 'RGB'
    else:
      raise ValueError('Only 1 or 3 samples per pixel supported, got', ns)
    self.mode = mode
    self.channel = 0
    pi = attrs['PhotometricInterpretation']
    if pi == 'MONOCHROME1':
      pass # Bright to dark values.
    if pi == 'MONOCHROME2':
      pass # Dark to bright values.

    ppv = attrs.get('PixelPaddingValue')
    if ppv is not None:
      self.pad_value = self.rescale_slope * ppv + self.rescale_intercept
    else:
      self.pad_value = None

    xsize, ysize = attrs['Columns'], attrs['Rows']

    self.data_size = (xsize, ysize, len(self.paths))
    xs, ys = [float(s) for s in attrs['PixelSpacing']]
    zs = series.z_plane_spacing()
    if zs is None:
      zs = 1 # Single plane image
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
    import pydicom
    d = pydicom.dcmread(self.paths[k])
    data = d.pixel_array
    if channel is not None:
      data = data[:,:,channel]
    return data

# -----------------------------------------------------------------------------
# PixelRepresentation 0 = unsigned, 1 = signed
#
def numpy_value_type(bits_allocated, pixel_representation, rescale_slope, rescale_intercept):

  from numpy import int8, uint8, int16, uint16, float32
  if (rescale_slope != 1 or
      int(rescale_intercept) != rescale_intercept or
      rescale_intercept < 0 and pixel_representation == 0):  # unsigned with negative offset
    return float32

  types = {(8,0): uint8,
           (8,1): int8,
           (16,0): uint16,
           (16,1): int16}
  if (bits_allocated, pixel_representation) in types:
    return types[(bits_allocated, pixel_representation)]

  raise ValueError('Unsupported value type, bits_allocated = ', bits_allocated)
