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
from collections import defaultdict
from numpy import cross, dot
from numpy import (
    array, concatenate, empty
    , float32
    , int8, int16, int32
    , uint8, uint16
)
import os
import weakref
from pydicom import dcmread
from typing import Any, Dict, TypeVar, Union

from chimerax.core.models import Model, Surface

from chimerax.map.volume import open_grids, Volume
from chimerax.map_data import MapFileFormat, GridData
from chimerax.map_data.readarray import allocate_array

try:
    import gdcm # noqa import used elsewhere
except ModuleNotFoundError:
    _has_gdcm = False
else:
    _has_gdcm = True

Path = TypeVar("Path", os.PathLike, str, bytes, None)

class MismatchedUIDError(Exception):
    pass

class DICOMVolume(Volume):
    pass

class Patient(Model):
    """A set of DICOM files that have the same Patient ID"""
    def __init__(self, session, pid):
        self.pid = pid
        self.session = session
        Model.__init__(self, 'Patient %s' % pid, session)
        self.studies = []

    def studies_from_files(self, files) -> None:
        studies = defaultdict(list)
        for f in files:
            if hasattr(f, 'StudyInstanceUID'):
                studies[f.StudyInstanceUID].append(f)
            else:
                studies["Unknown Study"].append(f)
        for key, files in studies.items():
            study = Study(self.session, key)
            study.series_from_files(files)
            self.studies.append(study)
            self.add(study) # add the study as a child model

    # @requires_gui
    def show_info():
        pass

    def __str__(self):
        return f"Patient {self.pid} with {len(self.studies)} studies"


class Study(Model):
    """A set of DICOM files that have the same Study Instance UID"""
    def __init__(self, session, uid):
        self.uid = uid
        self.session = session
        Model.__init__(self, 'Study %s' % uid, session)
        self.series = [] # regular images

    def series_from_files(self, files) -> None:
        files = self.filter_unreadable(files)
        series = defaultdict(list)
        for f in files:
            if hasattr(f, 'SeriesInstanceUID'):
                series[f.SeriesInstanceUID].append(f)
            else:
                series['Unknown Series'].append(f)
        for key, files in series.items():
            s = Series(self.session, files)
            self.series.append(s)
            self.add(s.to_models())

#        series.sort(key=lambda s: s.sort_key)
#        plane_ids = {s.plane_uids: s for s in series}
#        for s in series:
#            ref = s.ref_plane_uids
#            if ref and ref in plane_ids:
#                s.refers_to_series = plane_ids[ref]
        #models, msg = self.dicom_volumes(image_series)
        ## Open contour models for DICOM RT Structure Set series.
        #if contour_series:
        #    cmodels, cmsg = self.dicom_contours(contour_series)
        #    models += cmodels
        #    msg += '\n' + cmsg
        #    # TODO: Associate contour models with image data they were derived from.

        ## Warn about unrecognized series types.
        #if extra_series:
        #    snames = ', '.os.path.join('%s (%s)' % (s.name, s.dicom_class) for s in extra_series)
        #    self.session.logger.warning(
        #        'Can only handle images and contours, got %d other series types: %s'
        #        % (len(extra_series), snames)
        #    )

        #gmodels = self.group_models(self.paths, models)

    def filter_unreadable(self, files):
        if _has_gdcm:
            return files # PyDicom will use gdcm to read 16-bit lossless jpeg

        # Python Image Library cannot read 16-bit lossless jpeg.
        keep = []
        for f in files:
            if f.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.70' and f.attributes.get('BitsAllocated') == 16:
                warning = 'Could not read DICOM %s because Python Image Library cannot read 16-bit lossless jpeg ' \
                          'images. This functionality can be enabled by installing python-gdcm'
                self.session.logger.warning(warning % f.paths[0])
            else:
                keep.append(f)
        return keep

    # @requires_gui
    def show_info(self) -> None:
        pass
        # return DICOMMetadataTool(self)

    def __str__(self):
        return f"Study {self.uid} with {len(self.series)} series"

class DICOM:
    def __init__(self, session, data: Union[Path,list[Path]]):
        self.patients = []
        self.session = session
        self.logger = session.logger
        if type(data) is not list:
            self.paths = [data]
        else:
            self.paths = data
        self.find_dicom_series(self.paths)

    def __contains__(self, patient: str) -> bool:
        for patient in self.patients:
            if patient.pid == patient:
                return True
            return False

    @classmethod
    def from_paths(cls, session, path: Union[Path,list[Path]]):
        return cls(session, path)

    def open(self):
        return self.patients, ""

    def dicom_grids(self, paths, log=None) -> list[Any]:
        return [s._to_grids() for s in series]

    def find_dicom_series(
        self, paths, search_directories: bool = True, search_subdirectories: bool = True,
    ) -> None:
        """Look through directories to find dicom files (.dcm) and group the ones
        that belong to the same study and image series.  Also determine the order
        of the 2D images (one per file) in the 3D stack.  A series must be in a single
        directory.  If the same study and series is found in two directories, they
        are treated as two different series.
        """
        dfiles = self.files_by_directory(
            paths, search_directories=search_directories,
            search_subdirectories=search_subdirectories
        )
        self.session.logger.info(str(dfiles))
        nseries = len(dfiles)
        nfiles = sum(len(dpaths) for dpaths in dfiles.values())
        nsfiles = 0
        for dpaths in dfiles.values():
            nsfiles += len(dpaths)
            if self.logger:
                self.logger.status('Reading DICOM series %d of %d files in %d series' % (nsfiles, nfiles, nseries))
            patients = self.dicom_patients(dpaths)
            for patient in patients:
                self.patients.extend(patients)
                #if not patient in self:
                # self.patients.extend(patients)
                #else:
                # set patient id to something else
        self.session.logger.info(str(self.patients))
        #for study in self.patients:
        #    for series in study:
        #        series.find_reference_uid()


    def dicom_patients(self, paths) -> list[Patient]:
        """Group DICOM files into series"""
        series = defaultdict(list)
        patients = []
        for path in paths:
            d = dcmread(path)
            if hasattr(d, 'PatientID'):
                series[d.PatientID].append(d)
            else:
                series["Unknown Patient"].append(d)
        for key, series in series.items():
            patient = Patient(self.session, key)
            patient.studies_from_files(series)
            patients.append(patient)
        return patients

    def files_by_directory(
        self, paths, search_directories=True, search_subdirectories=True,
        suffix='.dcm', _dfiles=None
    ) -> Dict[str,list[str]]:
        """Find all dicom files (suffix .dcm) in directories and subdirectories
        and group them by directory"""
        dfiles = {} if _dfiles is None else _dfiles
        for p in paths:
            if os.path.isfile(p) and p.endswith(suffix):
                d = os.path.dirname(p)
                if d in dfiles:
                    dfiles[d].add(p)
                else:
                    dfiles[d] = set([p])
            elif search_directories and os.path.isdir(p):
                ppaths = [os.path.join(p, fname) for fname in os.listdir(p)]
                self.files_by_directory(
                    ppaths, search_directories=search_subdirectories,
                    search_subdirectories=search_subdirectories, _dfiles=dfiles
                )
        return dfiles


class DICOMMapFormat(MapFileFormat, DICOM):
    def __init__(self):
        MapFileFormat.__init__(
            self, 'DICOM image', 'dicom', ['dicom'], ['dcm'],
            batch=True, allow_directory=True
        )

    @property
    def open_func(self):
        return self.open_dicom_grids

    def open_dicom_grids(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        grids = self.dicom_grids(paths)
        return grids


class DicomContours(Model):
    def __init__(self, session, data):
        def rgb_255(cs):
            return tuple(int(c) for c in cs)

        def xyz_list(xs):
            a = tuple(float(x) for x in xs)
            xyz = array(a, float32).reshape(len(a) // 3, 3)
            return xyz

        if type(data) is not list:
            data = [data]
        self.dicom_series = data[0]

        if len(data) > 1:
            raise ValueError(
                'DICOM series has %d files, can only handle one file for "RT Structure Set Storage", '
                'file %s' % (len(series.paths), path)
        )
        desc = self.dicom_series.get('SeriesDescription', '')
        Model.__init__(self, 'Regions %s' % desc, session)

        el = self.dicom_elements(
            self.dicom_series
            , {
                'StructureSetROISequence': {
                    'ROINumber': int
                    , 'ROIName': str
                }
                , 'ROIContourSequence': {
                    'ROIDisplayColor': rgb_255
                    , 'ContourSequence': {
                        'ContourGeometricType':  str
                        , 'NumberOfContourPoints': int
                        , 'ContourData': xyz_list
                    }
                }
            }
        )
        regions = []
        for rs, rcs in zip(el['StructureSetROISequence'], el['ROIContourSequence']):
            r = ROIContourModel(session, rs['ROIName'], rs['ROINumber'], rcs['ROIDisplayColor'], rcs['ContourSequence'])
            regions.append(r)

        self.add(regions)

    def dicom_elements(self, data, fields):
        values = {}
        for name, v in fields.items():
            d = data.get(name)
            if d is None:
                raise ValueError('Did not find %s' % name)
            if isinstance(v, dict):
                values[name] = [self.dicom_elements(e, v) for e in d]
            else:
                values[name] = v(d)
        return values


class ROIContourModel(Surface):
    def __init__(self, session, name, number, color, contour_info):
        Model.__init__(self, name, session)
        self.roi_number = number
        opacity = 255
        self.color = tuple(color) + (opacity,)
        va, ta = self._contour_lines(contour_info)
        self.set_geometry(va, None, ta)
        self.display_style = self.Mesh
        self.use_lighting = False

    def _contour_lines(self, contour_info):
        points = []
        triangles = []
        nv = 0
        for ci in contour_info:
            ctype = ci['ContourGeometricType']
            if ctype != 'CLOSED_PLANAR':
                # TODO: handle other contour types
                continue
            np = ci['NumberOfContourPoints']  # noqa unused var
            pts = ci['ContourData']
            points.append(pts)
            n = len(pts)
            tri = empty((n, 2), int32)
            tri[:, 0] = tri[:, 1] = range(n)
            tri[:, 1] += 1
            tri[n - 1, 1] = 0
            tri += nv
            nv += n
            triangles.append(tri)
        va = concatenate(points)
        ta = concatenate(triangles)
        return va, ta


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

    def __init__(self, session, files):
        self.attributes = {}
        self.transfer_syntax = None
        self.files = files
        self._multiframe = None
        self._reverse_frames = False
        self._num_times = None
        self._z_spacing = None
        self.session = session
        self._log = session.logger
        self.image_series = True
        self.contour_series = False
        if any([f.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3' for f in files]):
            self.image_series = False
            self.contour_series = True
        #sample_file = files[0]
        #attrs = self.attributes
        #for attr in self.dicom_attributes:
        #    if hasattr(sample_file, attr):
        #        attrs[attr] = getattr(sample_file, attr)
        #for f in files:
        #    self.files.append(SeriesFile(f))
        #if self.transfer_syntax is None and hasattr(sample_file.file_meta, 'TransferSyntaxUID'):
        #    self.transfer_syntax = sample_file.file_meta.TransferSyntaxUID

    def to_models(self):
        if self.contour_series:
            return [DicomContours(self.session, s) for s in self.files]
        else:
            pass

#    def group_models(self, paths, models):
#        """Group into a four level hierarchy: directory, patient id, date, series."""
#        # Patient, Study, Series, Image
#        if len(models) == 0:
#            return []
#        dname = os.path.basename(paths[0]) if len(paths) == 1 else os.path.basename(os.path.dirname(paths[0]))
#        top = DICOMModel(dname, self.session)
#        locations = []
#        for m in models:
#            s = self.model_series(m)
#            if s is None:
#                locations.append((m, ()))
#            else:
#                pid = s.attributes.get('PatientID', 'unknown')
#                date = s.attributes.get('StudyDate', 'date unknown')
#                locations.append((m, ('Patient %s' % pid, date)))
#        leaf = {(): top}
#        for m, gnames in locations:
#            if gnames not in leaf:
#                for i in range(len(gnames)):
#                    if gnames[:i + 1] not in leaf:
#                        leaf[gnames[:i + 1]] = gm = DICOMModel(gnames[i], self.session)
#                        leaf[gnames[:i]].add([gm])
#            leaf[gnames].add([m])
#        return [top]

#    def model_series(self, m):
#        s = getattr(m, 'dicom_series', None)
#        if s is None:
#            # Look at child models for multi-channel and time-series.
#            for c in m.child_models():
#                s = getattr(c, 'dicom_series', None)
#                if s:
#                    break
#        return s

    @property
    def name(self):
        attrs = self.attributes
        fields = []
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

    def _to_grids(self, series) -> list['DicomGrid']:
        grids = []
        derived = []  # For grouping derived series with original series
        sgrids = {}
        for s in series:
            if not s.has_image_data:
                continue
            d = DicomData(s)
            if d.mode == 'RGB':
                # Create 3-channels for RGB series
                cgrids = []
                colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
                suffixes = [' red', ' green', ' blue']
                for channel in (0, 1, 2):
                    g = DicomGrid(d, channel=channel)
                    g.name += suffixes[channel]
                    g.rgba = colors[channel]
                    cgrids.append(g)
                grids.append(cgrids)
            elif s.num_times > 1:
                # Create time series for series containing multiple times as frames
                tgrids = []
                for t in range(s.num_times):
                    g = DicomGrid(d, time=t)
                    g.series_index = t
                    tgrids.append(g)
                grids.append(tgrids)
            else:
                # Create single channel, single time series.
                g = DicomGrid(d)
                rs = getattr(s, 'refers_to_series', None)
                if rs:
                    # If this associated with another series (e.g. is a segmentation), make
                    # it a channel together with that associated series.
                    derived.append((g, rs))
                else:
                    sgrids[s] = gg = [g]
                    grids.append(gg)
        # Group derived series with the original series
        channel_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        for g, rs in derived:
            sg = sgrids[rs]
            if len(sg) == 1:
                sg[0].channel = 1
            sg.append(g)
            g.channel = len(sg)
            g.rgba = channel_colors[(g.channel - 2) % len(channel_colors)]
            if not g.dicom_data.origin_specified:
                # Segmentation may not have specified an origin
                g.set_origin(sg[0].origin)
        # Show only first group of grids
        for gg in grids[1:]:
            for g in gg:
                g.show_on_open = False
        return grids

    @property
    def sort_key(self):
        attrs = self.attributes
        return (attrs.get('PatientID', ''), attrs.get('StudyDate', ''), self.name)

    @property
    def plane_uids(self):
        return tuple(fi._instance_uid for fi in self.files)

    @property
    def ref_plane_uids(self):
        fis = self.files
        if len(fis) == 1 and hasattr(fis[0], '_ref_instance_uids'):
            uids = fis[0]._ref_instance_uids
            if uids is not None:
                return tuple(uids)
        return None

    @property
    def num_times(self):
        if self._num_times is None:
            nt = self.attributes.get('NumberOfTemporalPositions', None)
            if nt is None:
                times = sorted(set(data.trigger_time for data in self.files))
                nt = len(times)
                for data in self.files:
                    data._time = times.index(data.trigger_time) + 1
                    data.inferred_properties += "TemporalPositionIdentifier"
                if nt > 1:
                    self._log.warning(
                        "Inferring time series from TriggerTime metadata \
                        field in series missing NumberOfTemporalPositions"
                    )
            else:
                nt = int(nt)
            self._num_times = nt
        return self._num_times

    @property
    def multiframe(self):
        mf = self._multiframe
        if mf is None:
            mf = False
            for fi in self.files:
                if fi.multiframe:
                    self._multiframe = mf = True
                    break
            self._multiframe = mf
        return mf

    def order_slices(self):
        files = self.paths
        if len(files) == 1 and self.multiframe:
            # Determination of whether frames reversed done in z_plane_spacing()
            self.z_plane_spacing()

        if len(files) <= 1:
            return

        # Check that time series images all have time value, and all times are found
        self._validate_time_series()

        files = self.files
        self._sort_by_z_position(files)

    def _validate_time_series(self):
        if self.num_times == 1:
            return

        files = self.files
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

        tcount = {t: 0 for t in tset}
        for fi in files:
            tcount[fi._time] += 1
        nz = len(files) / self.num_times
        for t, c in tcount.items():
            if c != nz:
                raise ValueError(
                    'DICOM time series time %d has %d images, expected %d'
                    % (t, c, nz)
                    )

    def grid_size(self):
        attrs = self.attributes
        xsize, ysize = attrs['Columns'], attrs['Rows']
        files = self.files
        if self.multiframe:
            if len(files) == 1:
                zsize = self.files[0]._num_frames
            else:
                maxf = max(fi._num_frames for fi in files)
                raise ValueError(
                    'DICOM multiple paths (%d), with multiple frames (%d) not supported, %s'
                    % (npaths, maxf, files[0].path)
                    )  # noqa npaths not defined
        else:
            zsize = len(files) // self.num_times

        return (xsize, ysize, zsize)

    def origin(self):
        files = self.files
        if len(files) == 0:
            return None

        pos = files[0]._position
        if pos is None:
            return None

        if self.multiframe and self._reverse_frames:
            zoffset = files[0]._num_frames * -self.z_plane_spacing()
            zaxis = self.plane_normal()
            pos = tuple(a + zoffset * b for a, b in zip(pos, zaxis))

        return pos

    def rotation(self):
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = self._patient_axes()
        return ((x0, x1, x2), (y0, y1, y2), (z0, z1, z2))

    def _patient_axes(self):
        files = self.files
        if files:
            # TODO: Different files can have different orientations.
            #   For example, study 02ef8f31ea86a45cfce6eb297c274598/series-000004.
            #   These should probably be separated into different series.
            orient = files[0]._orientation
            if orient is not None:
                x_axis, y_axis = orient[0:3], orient[3:6]
                z_axis = cross(x_axis, y_axis)
                return (x_axis, y_axis, z_axis)
        return ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def plane_normal(self):
        return self._patient_axes()[2]

    def _sort_by_z_position(self, series_files):
        z_axis = self.plane_normal()
        series_files.sort(key=lambda sf: (sf._time, (0 if sf._position is None else dot(sf._position, z_axis))))

    def pixel_spacing(self):
        pspacing = self.attributes.get('PixelSpacing')
        if pspacing is None and self.multiframe:
            pspacing = self.files[0]._pixel_spacing

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
                self._log.warning(
                    'Cannot determine z spacing, missing ImagePositionPatient, using value 1, %s'
                    % self.paths[0]
                    )
            zs = 1  # Single plane image
        elif zs == 0:
            if self._log:
                self._log.warning('Error. Image planes are at same z-position.  Setting spacing to 1.')
            zs = 1

        return (xs, ys, zs)

    def z_plane_spacing(self):
        dz = self._z_spacing
        if dz is None:
            files = self.files
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
                    z = [dot(fp, z_axis) for fp in fpos]
                    dz = self._spacing(z)
                if dz is not None and dz < 0:
                    self._reverse_frames = True
                    dz = abs(dz)
            elif len(files) < 2:
                dz = None
            else:
                nz = self.grid_size()[2]  # For time series just look at first time point.
                z_axis = self.plane_normal()
                z = [dot(f._position, z_axis) for f in files[:nz]]
                dz = self._spacing(z)
            self._z_spacing = dz
        return dz

    def _spacing(self, z):
        spacings = [(z1 - z0) for z0, z1 in zip(z[:-1], z[1:])]
        dzmin, dzmax = min(spacings), max(spacings)
        tolerance = 1e-3 * max(abs(dzmax), abs(dzmin))
        if dzmax - dzmin > tolerance:
            if self._log:
                msg = ('Plane z spacings are unequal, min = %.6g, max = %.6g, using max.\n' % (dzmin, dzmax) +
                       'Perpendicular axis (%.3f, %.3f, %.3f)\n' % tuple(self.plane_normal()) +
                       'Directory %s\n' % os.path.dirname(self.files[0].path) +
                       '\n'.join(['%s %s' % (os.path.basename(f.path), f._position) for f in self.files]))
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
    def __init__(self, data):
        self.data = data
        self.inferred_properties = []
        pos = getattr(data, 'ImagePositionPatient', None)
        self._position = tuple(float(p) for p in pos) if pos else None
        orient = getattr(data, 'ImageOrientationPatient', None)  # horz and vertical image axes
        self._orientation = tuple(float(p) for p in orient) if orient else None
        num = getattr(data, 'InstanceNumber', None)
        self._num = int(num) if num else None
        # TODO: Should this just be order and not time?
        t = getattr(data, 'TemporalPositionIdentifier', None)
        self._time = int(t) if t else None
        nf = getattr(data, 'NumberOfFrames', None)
        self._num_frames = int(nf) if nf is not None else None
        gfov = getattr(data, 'GridFrameOffsetVector', None)
        self._grid_frame_offset_vector = [float(o) for o in gfov] if gfov is not None else None
        self._class_uid = getattr(data, 'SOPClassUID', None)
        self._instance_uid = getattr(data, 'SOPInstanceUID', None)
        self._ref_instance_uid = getattr(data, 'ReferencedSOPInstanceUID', None)
        self._trigger_time = getattr(data, 'TriggerTime', None)
        self._pixel_spacing = None
        self._frame_positions = None
        if self._num_frames is not None:
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
                ,
                (('SharedFunctionalGroupsSequence', 1), ('DerivationImageSequence', 1), ('SourceImageSequence', 'all'))
                , 'ReferencedSOPInstanceUID'
            )

    def __lt__(self, im):
        if self._time == im._time:
            # Use z position instead of image number to assure right-handed coordinates.
            return self._position[2] < im._position[2]
        else:
            return self._time < im._time

    @property
    def trigger_time(self):
        return self._trigger_time

    @property
    def multiframe(self):
        nf = self._num_frames
        return nf is not None and nf > 1

    def _sequence_elements(self, data, seq_names, element_name, convert=None):
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


class DicomData:
    def __init__(self, series):
        self.dicom_series = series
        self.paths = tuple(series.paths)
        npaths = len(series.paths)  # noqa assigned but not accessed
        self.name = series.name
        attrs = series.attributes
        rsi = float(attrs.get('RescaleIntercept', 0))
        if rsi == int(rsi):
            rsi = int(rsi)
        self.rescale_intercept = rsi
        self.rescale_slope = float(attrs.get('RescaleSlope', 1))
        bits = attrs.get('BitsAllocated')
        rep = attrs.get('PixelRepresentation')
        self.value_type = self.numpy_value_type(bits, rep, self.rescale_slope, self.rescale_intercept)
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
            pass  # Bright to dark values.
        if pi == 'MONOCHROME2':
            pass  # Dark to bright values.
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
        dsize = self.data_size  # noqa assigned but not accessed
        if self.files_are_3d:
            # TODO:
            # data = None
            # if channel is not None:
            #     data = self.d.pixel_array[:,:,:,channel]
            # else:
            #     data = self.d.pixel_array
            a = self.read_frames(time, channel)
            array[:] = a[k0: k0 + ksz:kstep, j0:j0 + jsz:jstep, i0:i0 + isz:istep]
        else:
            for k in range(k0, k0 + ksz, kstep):
                if progress:
                    progress.plane((k - k0) // kstep)
                p = self.read_plane(k, time, channel, rescale=False)
                array[(k - k0) // kstep, :, :] = p[j0: j0 + jsz: jstep, i0:i0 + isz:istep]
        if self.rescale_slope != 1:
            array *= self.rescale_slope
        if self.rescale_intercept != 0:
            array += self.rescale_intercept
        return array

    def read_plane(self, k, time=None, channel=None, rescale=True):
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

    def read_frames(self, time=None, channel=None):
        d = dcmread(self.paths[0])
        data = d.pixel_array
        if channel is not None:
            data = data[:, :, :, channel]
        return data

    def numpy_value_type(self, bits_allocated, pixel_representation, rescale_slope, rescale_intercept):
        # PixelRepresentation 0 = unsigned, 1 = signed
        if (
                rescale_slope != 1 or
                int(rescale_intercept) != rescale_intercept or
                rescale_intercept < 0 and pixel_representation == 0
        ):  # unsigned with negative offset
            return float32

        types = {
            (1, 0):  uint8,
            (1, 1):  int8,
            (8, 0):  uint8,
            (8, 1):  int8,
            (16, 0): uint16,
            (16, 1): int16
        }
        if (bits_allocated, pixel_representation) in types:
            return types[(bits_allocated, pixel_representation)]
        raise ValueError('Unsupported value type, bits_allocated = %d' % bits_allocated)


class DicomGrid(GridData):
    initial_rendering_options = {
        'projection_mode':    '3d',
        'colormap_on_gpu':    True,
        'full_region_on_gpu': True
    }

    def __init__(self, d, time=None, channel=None):
        self.dicom_data = d
        GridData.__init__(
            self, d.data_size, d.value_type,
            d.data_origin, d.data_step, rotation=d.data_rotation,
            path=d.paths, name=d.name,
            file_type='dicom', time=time, channel=channel
            )
        if d.files_are_3d:
            # For fast access of multiple planes this reading method avoids
            # opening same dicom file many times to read planes.  50x faster in tests.
            self.read_matrix = self.dicom_read_matrix
        else:
            # For better caching if we read the whole plane, this method caches the
            # whole plane even if only part of the plane is needed.
            self.read_xy_plane = self.dicom_read_xy_plane
        self.multichannel = (channel is not None)
        self.initial_plane_display = True
        s = d.dicom_series
        if s.attributes.get('BitsAllocated') == 1 or s.dicom_class == 'Segmentation Storage':
            self.binary = True  # Use initial thresholds for binary segmentation
            self.initial_image_thresholds = [(0.5, 0), (1.5, 1)]
        else:
            self.initial_image_thresholds = [(-1000, 0.0), (300, 0.9), (3000, 1.0)]
        self.ignore_pad_value = d.pad_value

    # ---------------------------------------------------------------------------
    # If GridData.read_xy_plane() uses this method then whole planes are cached
    # even when a partial plane is requested.  The whole DICOM planes are always
    # read.  Caching them helps performance when say an xz-plane is being read.
    def dicom_read_xy_plane(self, k):
        c = self.channel if self.multichannel else None
        m = self.dicom_data.read_plane(k, self.time, c)
        return m

    # ---------------------------------------------------------------------------
    # If GridData.read_matrix() uses this method it only caches the actual requested
    # data.  For multiframe DICOM files this is much faster than reading separate
    # planes where the dicom data has to be opened once for each plane.
    # In a 512 x 512 x 235 test with 3d dicom segmentations this is 50x faster
    # than reading xy planes one at a time.
    def dicom_read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
        c = self.channel if self.multichannel else None
        self.dicom_data.read_matrix(
            ijk_origin, ijk_size, ijk_step,
            self.time, c, m, progress
            )
        return m
