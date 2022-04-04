# vim: set expandtab shiftwidth=4 softtabstop=4:
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
from os import listdir
from os.path import basename, dirname, isfile, isdir, join

from numpy import int8, uint8, int16, uint16, float32
from pydicom import dcmread

from chimerax.geometry import cross_product, inner_product

def find_dicom_series(paths, search_directories = True, search_subdirectories = True,
                      log = None, verbose = False):
    """Look through directories to find dicom files (.dcm) and group the ones
    that belong to the same study and image series.  Also determine the order
    of the 2D images (one per file) in the 3D stack.  A series must be in a single
    directory.  If the same study and series is found in two directories, they
    are treated as two different series.
    """
    dfiles = files_by_directory(paths, search_directories = search_directories,
                                search_subdirectories = search_subdirectories)
    nseries = len(dfiles)
    nfiles = sum(len(dpaths) for dpaths in dfiles.values())
    nsfiles = 0
    series = []
    for dpaths in dfiles.values():
        nsfiles += len(dpaths)
        log.status('Reading DICOM series %d of %d files in %d series' % (nsfiles, nfiles, nseries))
        series.extend(dicom_file_series(dpaths, log = log, verbose = verbose))

    # Include patient id in model name only if multiple patients found
    pids = set(s.attributes['PatientID'] for s in series if 'PatientID' in s.attributes)
    use_patient_id_in_name = unique_prefix_length(pids) if len(pids) > 1 else 0
    for s in series:
        s.use_patient_id_in_name = use_patient_id_in_name
    series.sort(key = lambda s: s.sort_key)
    find_reference_series(series)
    return series


def dicom_file_series(paths, log = None, verbose = False):
    """Group DICOM files into series"""
    series = {}
    for path in paths:
        d = dcmread(path)
        if hasattr(d, 'SeriesInstanceUID'):
            series_id = d.SeriesInstanceUID
            if series_id in series:
                s = series[series_id]
            else:
                series[series_id] = s = Series(log = log)
            s.add(path, d)
    series = tuple(series.values())
    for s in series:
        s.order_slices()
    if verbose and log:
        for s in series:
            path = s.paths[0]
            d = dcmread(path)
            log.info('Data set: %s\n%s\n%s\n' % (path, d.file_meta, d))
    return series


class Series:
    """Set of DICOM files (.dcm suffix) that have the same series unique identifier (UID)."""
    #
    # Code assumes each file for the same SeriesInstanceUID will have the same
    # value for these parameters.  So they are only read for the first file.
    # Not sure if this is a valid assumption.
    #
    dicom_attributes = ['BitsAllocated', 'BodyPartExamined', 'Columns', 'Modality',
                        'NumberOfTemporalPositions',
                        'PatientID', 'PhotometricInterpretation',
                        'PixelPaddingValue', 'PixelRepresentation', 'PixelSpacing',
                        'RescaleIntercept', 'RescaleSlope', 'Rows',
                        'SamplesPerPixel', 'SeriesDescription', 'SeriesInstanceUID', 'SeriesNumber',
                        'SOPClassUID', 'StudyDate']
    def __init__(self, log = None):
        self.paths = []
        self.attributes = {}
        self.transfer_syntax = None
        self._file_info = []
        self._multiframe = None
        self._reverse_frames = False
        self._num_times = None
        self._z_spacing = None
        self.use_patient_id_in_name = 0
        self._log = log

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

        # Get image encoding format
        if self.transfer_syntax is None and hasattr(data.file_meta, 'TransferSyntaxUID'):
            self.transfer_syntax = data.file_meta.TransferSyntaxUID

    @property
    def name(self):
        attrs = self.attributes
        fields = []
        # if self.use_patient_id_in_name and 'PatientID' in attrs:
        #     n = self.use_patient_id_in_name
        #     fields.append(attrs['PatientID'][:n])
        desc = attrs.get('SeriesDescription')
        if desc:
            fields.append(desc)
        else:
            if 'BodyPartExamined' in attrs:
                fields.append(attrs['BodyPartExamined'])
            if 'Modality' in attrs:
                fields.append(attrs['Modality'])
        if 'SeriesNumber' in attrs:
            fields.append(str(attrs['SeriesNumber']))
        # if 'StudyDate' in attrs:
        #     fields.append(attrs['StudyDate'])
        if len(fields) == 0:
            fields.append('unknown')
        name = ' '.join(fields)
        return name

    @property
    def sort_key(self):
        attrs = self.attributes
        return (attrs.get('PatientID',''), attrs.get('StudyDate',''), self.name, self.paths[0])

    @property
    def plane_uids(self):
        return tuple(fi._instance_uid for fi in self._file_info)

    @property
    def ref_plane_uids(self):
        fis = self._file_info
        if len(fis) == 1 and hasattr(fis[0], '_ref_instance_uids'):
            uids = fis[0]._ref_instance_uids
            if uids is not None:
                return tuple(uids)
        return None

    @property
    def num_times(self):
        if self._num_times is None:
            nt = int(self.attributes.get('NumberOfTemporalPositions', 1))
        else:
            nt = self._num_times
        return nt

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
        if len(paths) == 1 and self.multiframe:
            # Determination of whether frames reversed done in z_plane_spacing()
            self.z_plane_spacing()

        if len(paths) <= 1:
            return

        # Check that time series images all have time value, and all times are found
        self._validate_time_series()

        files = self._file_info
        self._sort_by_z_position(files)

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
            if self._log:
                msg = ('DICOM series header says it has %d times but %d found, %s... %d files.'
                       % (self.num_times, len(tset), files[0].path, len(files)))
                self._log.warning(msg)
            self._num_times = len(tset)

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

    def origin(self):
        files = self._file_info
        if len(files) == 0:
            return None

        pos = files[0]._position
        if pos is None:
            return None

        if self.multiframe and self._reverse_frames:
            zoffset = files[0]._num_frames * -self.z_plane_spacing()
            zaxis = self.plane_normal()
            pos = tuple(a+zoffset*b for a,b in zip(pos, zaxis))

        return pos

    def rotation(self):
        (x0,y0,z0),(x1,y1,z1),(x2,y2,z2) = self._patient_axes()
        return ((x0,x1,x2),(y0,y1,y2),(z0,z1,z2))

    def _patient_axes(self):
        files = self._file_info
        if files:
            # TODO: Different files can have different orientations.
            #   For example, study 02ef8f31ea86a45cfce6eb297c274598/series-000004.
            #   These should probably be separated into different series.
            orient = files[0]._orientation
            if orient is not None:
                x_axis, y_axis = orient[0:3], orient[3:6]
                z_axis = cross_product(x_axis, y_axis)
                return (x_axis, y_axis, z_axis)
        return ((1,0,0),(0,1,0),(0,0,1))

    def plane_normal(self):
        return self._patient_axes()[2]

    def _sort_by_z_position(self, series_files):
        z_axis = self.plane_normal()
        series_files.sort(key = lambda sf: (sf._time, (0 if sf._position is None else inner_product(sf._position, z_axis))))

    def pixel_spacing(self):
        pspacing = self.attributes.get('PixelSpacing')
        if pspacing is None and self.multiframe:
            pspacing = self._file_info[0]._pixel_spacing

        if pspacing is None:
            xs = ys = 1
            if self._log:
                self._log.warning('Missing PixelSpacing, using value 1, %s' % self.paths[0])
        else:
            xs, ys = [float(s) for s in pspacing]
        zs = self.z_plane_spacing()
        if zs is None:
            nz = self.grid_size()[2]
            if nz > 1 and self._log:
                self._log.warning('Cannot determine z spacing, missing ImagePositionPatient, using value 1, %s'
                                  % self.paths[0])
            zs = 1 # Single plane image
        elif zs == 0:
            if self._log:
                self._log.warning('Error. Image planes are at same z-position.  Setting spacing to 1.')
            zs = 1

        return (xs,ys,zs)

    def z_plane_spacing(self):
        dz = self._z_spacing
        if dz is None:
            files = self._file_info
            if self.multiframe:
                f = files[0]
                fpos = f._frame_positions
                if fpos is None:
                    gfov = f._grid_frame_offset_vector
                    if gfov is None:
                        dz = None
                    else:
                        dz = self._spacing(gfov)
                else:
                    # TODO: Need to reverse order if z decrease as frame number increases
                    z_axis = self.plane_normal()
                    z = [inner_product(fp, z_axis) for fp in fpos]
                    dz = self._spacing(z)
                if dz is not None and dz < 0:
                    self._reverse_frames = True
                    dz = abs(dz)
            elif len(files) < 2:
                dz = None
            else:
                nz = self.grid_size()[2]  # For time series just look at first time point.
                z_axis = self.plane_normal()
                z = [inner_product(f._position, z_axis) for f in files[:nz]]
                dz = self._spacing(z)
            self._z_spacing = dz
        return dz

    def _spacing(self, z):
        spacings = [(z1-z0) for z0,z1 in zip(z[:-1],z[1:])]
        dzmin, dzmax = min(spacings), max(spacings)
        tolerance = 1e-3 * max(abs(dzmax), abs(dzmin))
        if dzmax-dzmin > tolerance:
            if self._log:
                msg = ('Plane z spacings are unequal, min = %.6g, max = %.6g, using max.\n' % (dzmin, dzmax) +
                       'Perpendicular axis (%.3f, %.3f, %.3f)\n' % tuple(self.plane_normal()) +
                       'Directory %s\n' % dirname(self._file_info[0].path) +
                       '\n'.join(['%s %s' % (basename(f.path), f._position) for f in self._file_info]))
                self._log.warning(msg)
        dz = dzmax if abs(dzmax) > abs(dzmin) else dzmin
        return dz

    @property
    def has_image_data(self):
        attrs = self.attributes
        for attr in ('BitsAllocated', 'PixelRepresentation'):
            if attrs.get(attr) is None:
                return False
        return True

    @property
    def dicom_class(self):
        cuid = self.attributes.get('SOPClassUID')
        return 'unknown' if cuid is None else cuid.name


class SeriesFile:
    def __init__(self, path, data):
        self.path = path
        pos = getattr(data, 'ImagePositionPatient', None)
        self._position = tuple(float(p) for p in pos) if pos else None
        orient = getattr(data, 'ImageOrientationPatient', None)	# horz and vertical image axes
        self._orientation = tuple(float(p) for p in orient) if orient else None
        num = getattr(data, 'InstanceNumber', None)
        self._num = int(num) if num else None
        t = getattr(data, 'TemporalPositionIdentifier', None)
        self._time = int(t) if t else None
        nf = getattr(data, 'NumberOfFrames', None)
        self._num_frames = int(nf) if nf is not None else None
        gfov = getattr(data, 'GridFrameOffsetVector', None)
        self._grid_frame_offset_vector = [float(o) for o in gfov] if gfov is not None else None
        cuid = getattr(data, 'SOPClassUID', None)
        self._class_uid = cuid
        inst = getattr(data, 'SOPInstanceUID', None)
        self._instance_uid = inst
        ref = getattr(data, 'ReferencedSOPInstanceUID', None)
        self._ref_instance_uid = ref
        self._pixel_spacing = None
        self._frame_positions = None
        if nf is not None:
            def floats(s):
                return [float(x) for x in s]
            self._pixel_spacing = self._sequence_elements(
                data
                , (('SharedFunctionalGroupsSequence', 1), ('PixelMeasuresSequence', 1))
                , 'PixelSpacing'
                , floats
            )
            self._frame_positions = self._sequence_elements(
                data
                , (('PerFrameFunctionalGroupsSequence', 'all'), ('PlanePositionSequence', 1))
                , 'ImagePositionPatient'
                , floats
            )
            self._ref_instance_uids = self._sequence_elements(
                data
                , (('SharedFunctionalGroupsSequence', 1), ('DerivationImageSequence', 1), ('SourceImageSequence', 'all'))
                , 'ReferencedSOPInstanceUID'
            )

    def __lt__(self, im):
        if self._time == im._time:
            # Use z position instead of image number to assure right-handed coordinates.
            return self._position[2] < im._position[2]
        else:
            return self._time < im._time

    @property
    def multiframe(self):
        nf = self._num_frames
        return nf is not None and nf > 1

    def _sequence_elements(self, data, seq_names, element_name, convert = None):
        if len(seq_names) == 0:
            value = getattr(data, element_name, None)
            if value is not None and convert is not None:
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


def files_by_directory(paths, search_directories = True, search_subdirectories = True,
                       suffix = '.dcm', _dfiles = None):
    """Find all dicom files (suffix .dcm) in directories and subdirectories
    and group them by directory"""
    dfiles = {} if _dfiles is None else _dfiles
    for p in paths:
        if isfile(p) and p.endswith(suffix):
            d = dirname(p)
            if d in dfiles:
                dfiles[d].add(p)
            else:
                dfiles[d] = set([p])
        elif search_directories and isdir(p):
            ppaths = [join(p, fname) for fname in listdir(p)]
            files_by_directory(ppaths, search_directories=search_subdirectories,
                               search_subdirectories=search_subdirectories, _dfiles=dfiles)
    return dfiles


class DicomData:
    def __init__(self, series):
        self.dicom_series = series
        self.paths = tuple(series.paths)
        npaths = len(series.paths) # noqa assigned but not accessed
        self.name = series.name
        attrs = series.attributes
        rsi = float(attrs.get('RescaleIntercept', 0))
        if rsi == int(rsi):
            rsi = int(rsi)
        self.rescale_intercept = rsi
        self.rescale_slope = float(attrs.get('RescaleSlope', 1))
        bits = attrs.get('BitsAllocated')
        rep = attrs.get('PixelRepresentation')
        self.value_type = numpy_value_type(bits, rep, self.rescale_slope, self.rescale_intercept)
        ns = attrs.get('SamplesPerPixel')
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
        self.files_are_3d = series.multiframe
        self._reverse_planes = (series.multiframe and series._reverse_frames)
        self.data_size = series.grid_size()
        self.data_step = series.pixel_spacing()
        self.data_origin = origin = series.origin()
        if origin is None:
            self.origin_specified = False
            self.data_origin = (0, 0, 0)
        else:
            self.origin_specified = True
        self.data_rotation = series.rotation()

    def read_matrix(self, ijk_origin, ijk_size, ijk_step, time, channel, array, progress):
        """Reads a submatrix and returns 3D NumPy matrix with zyx index order."""
        i0, j0, k0 = ijk_origin
        isz, jsz, ksz = ijk_size
        istep, jstep, kstep = ijk_step
        dsize = self.data_size # noqa assigned but not accessed
        if self.files_are_3d:
            a = self.read_frames(time, channel)
            array[:] = a[k0: k0 + ksz:kstep, j0:j0 + jsz:jstep, i0:i0 + isz:istep]
        else:
            for k in range(k0, k0 + ksz, kstep):
                if progress:
                    progress.plane((k - k0) // kstep)
                p = self.read_plane(k, time, channel, rescale = False)
                array[(k - k0) // kstep, :, :] = p[j0: j0 + jsz: jstep, i0:i0 + isz:istep]
        if self.rescale_slope != 1:
            array *= self.rescale_slope
        if self.rescale_intercept != 0:
            array += self.rescale_intercept
        return array

    def read_plane(self, k, time = None, channel = None, rescale = True):
        if self._reverse_planes:
            klast = self.data_size[2] - 1
            k = klast - k
        if self.files_are_3d:
            d = dcmread(self.paths[0])
            data = d.pixel_array[k]
        else:
            p = k if time is None else (k + self.data_size[2] * time)
            d = dcmread(self.paths[p])
            data = d.pixel_array
        if channel is not None:
            data = data[:, :, channel]
        a = data.astype(self.value_type) if data.dtype != self.value_type else data
        if rescale:
            if self.rescale_slope != 1:
                a *= self.rescale_slope
            if self.rescale_intercept != 0:
                a += self.rescale_intercept
        return a

    def read_frames(self, time = None, channel = None):
        d = dcmread(self.paths[0])
        data = d.pixel_array
        if channel is not None:
            data = data[:, :, :, channel]
        return data


def numpy_value_type(bits_allocated, pixel_representation, rescale_slope, rescale_intercept):
    # PixelRepresentation 0 = unsigned, 1 = signed
    if (
        rescale_slope != 1 or
        int(rescale_intercept) != rescale_intercept or
        rescale_intercept < 0 and pixel_representation == 0
    ):  # unsigned with negative offset
        return float32

    types = {
      (1, 0): uint8,
      (1, 1): int8,
      (8, 0): uint8,
      (8, 1): int8,
      (16, 0): uint16,
      (16, 1): int16
    }
    if (bits_allocated, pixel_representation) in types:
        return types[(bits_allocated, pixel_representation)]
    raise ValueError('Unsupported value type, bits_allocated = %d' % bits_allocated)

def unique_prefix_length(strings):
    sset = set(strings)
    maxlen = max(len(s) for s in sset)
    for i in range(maxlen):
        if len(set(s[:i] for s in sset)) == len(sset):
            return i
    return maxlen

def find_reference_series(series):
    plane_ids = {s.plane_uids: s for s in series}
    for s in series:
        ref = s.ref_plane_uids
        if ref and ref in plane_ids:
            s.refers_to_series = plane_ids[ref]
