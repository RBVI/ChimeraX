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

  # Include patient id in model name only if multiple patients found
  pids = set(s.attributes['PatientID'] for s in series if 'PatientID' in s.attributes)
  use_patient_id_in_name = unique_prefix_length(pids) if len(pids) > 1 else 0
  for s in series:
    s.use_patient_id_in_name = use_patient_id_in_name
    
  series.sort(key = lambda s: (s.name, s.paths[0]))
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
      s.add(path, d)

  series = tuple(series.values())
  for s in series:
    s.order_slices()

  if verbose:
    for s in series:
      path = s.paths[0]
      d = dcmread(path)
      print ('Data set: %s\n%s\n' % (path, d))

  return series

# -----------------------------------------------------------------------------
# Set of dicom files (.dcm suffix) that have the same series unique identifer (UID).
#
class Series:
  #
  # Code assumes each file for the same SeriesInstanceUID will have the same
  # value for these parameters.  So they are only read for the first file.
  # Not sure if this is a valid assumption.
  #
  dicom_attributes = ['BitsAllocated', 'Columns', 'Modality',
                      'NumberOfTemporalPositions',
                      'PatientID', 'PhotometricInterpretation',
                      'PixelPaddingValue', 'PixelRepresentation', 'PixelSpacing',
                      'RescaleIntercept', 'RescaleSlope', 'Rows',
                      'SamplesPerPixel', 'SeriesDescription', 'SeriesInstanceUID',
                      'StudyDate']
  def __init__(self):
    self.paths = []
    self.attributes = {}
    self._file_info = []
    self._multiframe = None
    self.use_patient_id_in_name = 0
    
  def add(self, path, data):
    # Read attributes that should be the same for all planes.
    if len(self.paths) == 0:
      attrs = self.attributes
      for attr in self.dicom_attributes:
        if hasattr(data, attr):
          attrs[attr] = getattr(data, attr)

    self.paths.append(path)

    # Read attributes used for ordering the images.
    self._file_info.append(SeriesFile(path, data))

  @property
  def name(self):
    attrs = self.attributes
    fields = []
    if self.use_patient_id_in_name and 'PatientID' in attrs:
      n = self.use_patient_id_in_name
      fields.append(attrs['PatientID'][:n])
    if 'SeriesDescription' in attrs:
      fields.append(attrs['SeriesDescription'])
    if 'StudyDate' in attrs:
      fields.append(attrs['StudyDate'])
    name = ' '.join(fields)
    return name

  def _dump_data(self, data):
    for elem in data:
      print(elem)
      if elem.VR == 'SQ':
        print ('seq element')
        for ds in elem:
          for e2 in ds:
            print('>>>   ', e2)

  @property
  def num_times(self):
    return int(self.attributes.get('NumberOfTemporalPositions', 1))

  @property
  def multiframe(self):
    mf = self._multiframe
    if mf is None:
      mf = False
      for fi in self._file_info:
        if fi.multiframe:
          self._multiframe = mf = True
          break
      self._multiframe = mf
    return mf

  def order_slices(self):
    paths = self.paths
    if len(paths) <= 1:
      return paths

    # Check that all images have an instance number for ordering.
    files = self._file_info
    for fi in files:
      if fi._num is None:
        raise ValueError("Missing dicom InstanceNumber, can't order slice %s" % fi.path)

    # Check that time series images all have time value, and all times are found
    self._validate_time_series()

    files.sort()

    self.paths = tuple(fi.path for fi in files)

  def _validate_time_series(self):
    if self.num_times == 1:
      return

    files = self._file_info
    for fi in files:
      if fi._time is None:
        raise ValueError('Missing dicom TemporalPositionIdentifier for image %s' % fi.path)

    tset = set(fi._time for fi in files)
    if len(tset) != self.num_times:
      raise ValueError('DICOM series says it has %d times but %d found, %s... %d files.'
                       % (self.num_times, len(tset), files[0].path, len(files)))

    tcount = {t:0 for t in tset}
    for fi in files:
      tcount[fi._time] += 1
    nz = len(files) / self.num_times
    for t,c in tcount.items():
      if c != nz:
        raise ValueError('DICOM time series time %d has %d images, expected %d'
                         % (t, c, nz))

  def grid_size(self):

    attrs = self.attributes
    xsize, ysize = attrs['Columns'], attrs['Rows']
    files = self._file_info
    if self.multiframe:
      if len(files) == 1:
        zsize = self._file_info[0]._num_frames
      else:
        maxf = max(fi._num_frames for fi in files)
        raise ValueError('DICOM multiple paths (%d), with multiple frames (%d) not supported, %s'
                         % (npaths, maxf, files[0].path))
    else:
      zsize = len(files) // self.num_times

    return (xsize, ysize, zsize)

  def pixel_spacing(self):

    if self.multiframe:
      pspacing = self._file_info[0]._pixel_spacing
    else:
      pspacing = self.attributes.get('PixelSpacing')

    if pspacing is None:
      xs = ys = 1
      print('Missing PixelSpacing, using value 1, %s' % self.paths[0])
    else:
      xs, ys = [float(s) for s in pspacing]
    zs = self.z_plane_spacing()
    if zs is None:
      if len(self._file_info) > 1:
        print('Cannot determine z spacing, missing ImagePositionPatient, using value 1, %s'
              % self.paths[0])
      zs = 1 # Single plane image

    return (xs,ys,zs)

  def z_plane_spacing(self):
    files = self._file_info
    if self.multiframe:
      fpos = files[0]._frame_positions
      if fpos is None:
        dz = None
      else:
        # TODO: Need to reverse order if z decrease as frame number increases
        dz = fpos[1][2] - fpos[0][2]
    elif len(files) < 2:
      dz = None
    else:
      dz = files[1]._position[2] - files[0]._position[2]
    return dz

# -----------------------------------------------------------------------------
#
class SeriesFile:
  def __init__(self, path, data):
    self.path = path

    pos = getattr(data, 'ImagePositionPatient', None)
    self._position = tuple(float(p) for p in pos) if pos else None
    
    num = getattr(data, 'InstanceNumber', None)
    self._num = int(num) if num else None

    t = getattr(data, 'TemporalPositionIdentifier', None)
    self._time = int(t) if t else None

    nf = getattr(data, 'NumberOfFrames', None)
    self._num_frames = int(nf) if nf is not None else None

    self._pixel_spacing = None
    self._frame_positions = None
    if nf is not None:
      def floats(s):
        return [float(x) for x in s]
      self._pixel_spacing = self._sequence_elements(data,
                                                   (('SharedFunctionalGroupsSequence', 1),
                                                    ('PixelMeasuresSequence', 1)),
                                                   'PixelSpacing', floats)
      self._frame_positions = self._sequence_elements(data,
                                                     (('PerFrameFunctionalGroupsSequence', 'all'),
                                                      ('PlanePositionSequence', 1)),
                                                     'ImagePositionPatient', floats)

  def __lt__(self, im):
    if  self._time == im._time:
      # Use z position instead of image number to assure right-handed coordinates.
      return self._position[2] < im._position[2]
    else:
      return self._time < im._time

  @property
  def multiframe(self):
    nf = self._num_frames
    return nf is not None and nf > 1

  def _sequence_elements(self, data, seq_names, element_name, convert):
    if len(seq_names) == 0:
      value = getattr(data, element_name, None)
      if value is not None:
        value = convert(value)
      return value
    else:
      name, count = seq_names[0]
      seq = getattr(data, name, None)
      if seq is None:
        return None
      if count == 'all':
        values = [self._sequence_elements(e, seq_names[1:], element_name, convert)
                  for e in seq]
      else:
        values = self._sequence_elements(seq[0], seq_names[1:], element_name, convert)
      return values

# Old stuff
    sfgs = getattr(data, 'SharedFunctionalGroupsSequence', None)
    if sfgs is not None:
      sfgds = sfgs[0]
      pms = getattr(sfgds, 'PixelMeasuresSequence', None)
      if pms is not None:
        ps = getattr(pms[0], 'PixelSpacing', None)
        if ps is not None:
          print('pixel spacing', ps)

    pffgs = getattr(data, 'PerFrameFunctionalGroupsSequence', None)
    if pffgs is not None:
      for f,ffg in enumerate(pffgs):  # one item for each frame
        pps = getattr(ffg, 'PlanePositionSequence', None)
        if pps is not None:
          ipp = getattr(pps[0], 'ImagePositionPatient', None)
          if ipp is not None:
            print('Frame', f, 'position', ipp)


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
    npaths = len(series.paths)

    self.name = series.name

    attrs = series.attributes
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
      raise ValueError('Only 1 or 3 samples per pixel supported, got %d' % ns)
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

    self._files_are_3d = series.multiframe
    self.data_size = series.grid_size()
    self.data_step = series.pixel_spacing()
    self.data_origin = (0.0, 0.0, 0.0)

  # ---------------------------------------------------------------------------
  # Reads a submatrix and returns 3D NumPy matrix with zyx index order.
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, time, channel, array, progress):

    i0, j0, k0 = ijk_origin
    isz, jsz, ksz = ijk_size
    istep, jstep, kstep = ijk_step
    dsize = self.data_size
    if self._files_are_3d:
      a = self.read_frames(time, channel)
      array[:] = a[k0:k0+ksz:kstep,j0:j0+jsz:jstep,i0:i0+isz:istep]
    else:
      for k in range(k0, k0+ksz, kstep):
        if progress:
          progress.plane((k-k0)//kstep)
        p = self.read_plane(k, time, channel)
        array[(k-k0)//kstep,:,:] = p[j0:j0+jsz:jstep,i0:i0+isz:istep]

    if self.rescale_slope != 1:
      array *= self.rescale_slope
    if self.rescale_intercept != 0:
      array += self.rescale_intercept
    return array

  # ---------------------------------------------------------------------------
  #
  def read_plane(self, k, time = None, channel = None):
    p = k if time is None else (k + self.data_size[2]*time)
    import pydicom
    d = pydicom.dcmread(self.paths[p])
    data = d.pixel_array
    if channel is not None:
      data = data[:,:,channel]
    return data

  # ---------------------------------------------------------------------------
  #
  def read_frames(self, time = None, channel = None):
    import pydicom
    d = pydicom.dcmread(self.paths[0])
    data = d.pixel_array
    if channel is not None:
      data = data[:,:,:,channel]
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

  types = {(1,0): uint8,
           (1,1): int8,
           (8,0): uint8,
           (8,1): int8,
           (16,0): uint16,
           (16,1): int16}
  if (bits_allocated, pixel_representation) in types:
    return types[(bits_allocated, pixel_representation)]

  raise ValueError('Unsupported value type, bits_allocated = %d' % bits_allocated)

# -----------------------------------------------------------------------------
#
def unique_prefix_length(strings):
  sset = set(strings)
  maxlen = max(len(s) for s in sset)
  for i in range(maxlen):
    if len(set(s[:i] for s in sset)) == len(sset):
      return i
  return maxlen
