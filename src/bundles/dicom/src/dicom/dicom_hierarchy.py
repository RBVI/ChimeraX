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
import datetime
import os
from numpy import (
    cross, dot, float32
    , uint8, int8, uint16, int16
)

from pydicom import dcmread
from typing import Optional

from chimerax.core.decorators import requires_gui
from chimerax.core.models import Model
from chimerax.core.tools import get_singleton
from chimerax.core.session import Session

from chimerax.map.volume import open_grids

from .errors import MismatchedUIDError, UnrenderableSeriesError
from .dicom_models import DicomContours, DicomGrid

try:
    import gdcm # noqa import used elsewhere
except ModuleNotFoundError:
    _has_gdcm = False
else:
    _has_gdcm = True

class Patient(Model):
    """A set of DICOM files that have the same Patient ID"""
    def __init__(self, session: Session, pid: str):
        self.session = session
        self.pid = pid
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
            study = Study(self.session, key, self)
            study.series_from_files(files)
            self.studies.append(study)
        if studies:
            self.name = f'Patient (ID: {self.patient_id})'

    @requires_gui
    def show_info(self):
        from .ui import DICOMBrowserTool
        tool = get_singleton(self.session, DICOMBrowserTool, "DICOM Browser")
        tool.display(True)

    def merge_and_delete_other(self, other):
        if not other.pid == self.pid:
            raise MismatchedUIDError("Can't merge patients that don't have the same Patient ID")
        studies = defaultdict(list)
        for study in self.studies:
            studies[study.uid].append(study)
        for study in other.studies:
            studies[study.uid].append(study)
        for study_list in studies.values():
            ref_study = study_list[0]
            for study in study_list[1:]:
                ref_study.merge_and_delete_other(study)
            studies[ref_study.uid] = ref_study
        self.studies = list(studies.values())
        del other

    @property
    def birth_date_as_datetime(self):
        if self.birth_date:
            return datetime.datetime.strptime(
                self.birth_date, '%Y%m%d'
            )
        else:
            return None

    @property
    def birth_date(self):
        if self.studies:
            return self.studies[0].birth_date

    @property
    def patient_name(self):
        if self.studies:
            return self.studies[0].patient_name

    @property
    def patient_sex(self):
        if self.studies:
            return self.studies[0].patient_sex

    @property
    def patient_id(self):
        if self.studies:
            return self.studies[0].patient_id

    def __str__(self):
        return f"Patient {self.pid} with {len(self.studies)} studies"

    def render(self):
        try:
            self.session.models.add([self])
        except ValueError: # Already in scene
            pass
        for study in self.studies:
            self.add([study])
            study.series.sort(key = lambda x: x.number)
            study.open_series_as_models()

    def __iter__(self):
        return iter(self.studies)


class Study(Model):
    """A set of DICOM files that have the same Study Instance UID"""
    def __init__(self, session, uid, patient: Patient):
        self.uid = uid
        self.session = session
        self.patient = patient
        Model.__init__(self, 'Study (%s)' % uid, session)
        self.series = []  # regular images
        self._drawn_series = set()

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
        if self.series:
            if self.date_as_datetime:
                self.name = '%s Study (%s)' % (self.body_part, self.date_as_datetime.strftime("%Y-%m-%d"))
            else:
                self.name = '%s Study (Unknown Date)' % (self.body_part)
            self.series.sort(key=lambda s: s.sort_key)
            plane_ids = {s.plane_uids: s for s in self.series}
            for s in self.series:
                ref = s.ref_plane_uids
                if ref and ref in plane_ids:
                    s.refers_to_series = plane_ids[ref]

    def filter_unreadable(self, files):
        if _has_gdcm:
            return files  # PyDicom will use gdcm to read 16-bit lossless jpeg

        # Python Image Library cannot read 16-bit lossless jpeg.
        keep = []
        for f in files:
            if f.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.70' and f.get('BitsAllocated') == 16:
                warning = 'Could not read DICOM %s because Python Image Library cannot read 16-bit lossless jpeg ' \
                          'images. This functionality can be enabled by installing python-gdcm'
                self.session.logger.warning(warning % f.filename)
            else:
                keep.append(f)
        return keep

    @requires_gui
    def show_info(self):
        from .ui import DICOMBrowserTool
        tool = get_singleton(self.session, DICOMBrowserTool, "DICOM Browser")
        tool.display(True)

    def merge_and_delete_other(self, other):
        if not other.uid == self.uid:
            raise MismatchedUIDError("Can't merge studies that don't have the same Study UID")
        # TODO: Improve the logic here
        series = {}
        for series_ in self.series:
            series[series_.uid] = series_
        for series_ in other.series:
            if series_.uid not in series:
                series[series_.uid] = series_
        self.series = list(series.values())
        if self._child_drawings and other._child_drawings:
            self.add(other._child_drawings)
        other.delete()
        del other

    def open_series_as_models(self):
        if self.series:
            for s in self.series:
                try:
                    if s.uid not in self._drawn_series:
                        self.add(s.to_models())
                        self._drawn_series.add(s.uid)
                except UnrenderableSeriesError as e:
                    self.session.logger.warning(str(e))

    def __str__(self):
        return f"Study {self.uid} with {len(self.series)} series"

    @property
    def body_part(self):
        return self.series[0].body_part

    @property
    def birth_date(self):
        return self.series[0].birth_date

    @property
    def patient_name(self):
        return self.series[0].patient_name

    @property
    def patient_id(self):
        if self.series:
            return self.series[0].patient_id
        else:
            return "Unknown"

    @property
    def patient_sex(self):
        return self.series[0].patient_sex

    @property
    def date_as_datetime(self):
        if self.series[0].study_date:
            return datetime.datetime.strptime(
                self.series[0].study_date, '%Y%m%d'
            )
        else:
            return None

    @property
    def description(self):
        return self.series[0].study_description

    @property
    def study_date(self):
        return self.series[0].study_date

    @property
    def study_id(self):
        return self.series[0].study_id

    def __iter__(self):
        return iter(self.series)


class Series:
    """Set of DICOM files (.dcm suffix) that have the same series unique identifier (UID)."""
    #
    # Code assumes each file for the same SeriesInstanceUID will have the same
    # value for these parameters.  So they are only read for the first file.
    # Not sure if this is a valid assumption.
    #
    def __init__(self, session, files):
        self._raw_files = files
        self.order_slices()
        self.paths = [file.filename for file in self._raw_files]
        self.files = []
        for f in self._raw_files:
            self.files.append(SeriesFile(f))
        self.transfer_syntax = None
        self._multiframe = None
        self._reverse_frames = False
        self._num_times = None
        self._z_spacing = None
        self.session = session
        self.image_series = True
        self.contour_series = False
        if any([f.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3' for f in files]):
            self.image_series = False
            self.contour_series = True
        if not any([f.get("PixelData") for f in files]):
            self.image_series = False
        self.sample_file = self._raw_files[0]
        if self.transfer_syntax is None and hasattr(self.sample_file.file_meta, 'TransferSyntaxUID'):
            self.transfer_syntax = self.sample_file.file_meta.TransferSyntaxUID
        if len(files) == 1 and self.multiframe:
            # Determination of whether frames reversed done in z_plane_spacing()
            self.z_plane_spacing()
        # Check that time series images all have time value, and all times are found
        self._validate_time_series()

    def to_models(self):
        if self.contour_series:
            return [DicomContours(self.session, s, self.name) for s in self._raw_files]
        elif self.image_series:
            return open_grids(self.session, self._to_grids(), name=self.name)[0]
        else:
            raise UnrenderableSeriesError("No model created for Series #%s from patient %s because "
                                          "it had no pixel data. Metadata will still "
                                          "be available." % (self.number, self.patient_id))

    @property
    def name(self):
        return f"{self.number} {self.modality} ({self.description})"
    
    @property
    def sop_class_uid(self):
        return self.sample_file.get("SOPClassUID")

    @property
    def samples_per_pixel(self):
        return self.sample_file.get('SamplesPerPixel')

    @property
    def pixel_padding(self):
        return self.sample_file.get('PixelPaddingValue')

    @property
    def bits_allocated(self):
        return self.sample_file.get('BitsAllocated')

    @property
    def pixel_representation(self):
        return self.sample_file.get('PixelRepresentation')

    @property
    def photometric_interpretation(self):
        return self.sample_file.get('PhotometricInterpretation')

    @property
    def rescale_slope(self):
        return self.sample_file.get('RescaleSlope', 1)

    @property
    def rescale_intercept(self):
        return self.sample_file.get('RescaleIntercept', 0)

    @property
    def body_part(self):
        return self.sample_file.get("BodyPartExamined")

    @property
    def birth_date(self):
        return self.sample_file.get("PatientBirthDate")

    @property
    def patient_name(self):
        return self.sample_file.get("PatientName")

    @property
    def patient_id(self):
        return self.sample_file.get("PatientID", "")

    @property
    def patient_sex(self):
        return self.sample_file.get("PatientSex")

    @property
    def study_date(self):
        return self.sample_file.get("StudyDate", "")

    @property
    def study_id(self):
        return self.sample_file.get("StudyID")

    @property
    def study_description(self):
        return self.sample_file.get("StudyDescription")

    @property
    def number(self):
        return self.sample_file.get("SeriesNumber", "Unknown Series Number")

    @property
    def description(self):
        return self.sample_file.get("SeriesDescription", "No Description")

    @property
    def modality(self):
        return self.sample_file.get("Modality", "Unknown Modality")

    @property
    def uid(self):
        return self.sample_file.get("SeriesInstanceUID")

    @property
    def columns(self):
        return self.sample_file.get("Columns")

    @property
    def rows(self):
        return self.sample_file.get("Rows")

    @property
    def size(self) -> Optional[str]:
        x, y = self.columns, self.rows
        if x and y:
            return "%sx%s" % (x, y)
        return None

    def _to_grids(self) -> list['DicomGrid']:
        grids = []
        derived = []  # For grouping derived series with original series
        sgrids = {}
        d = DicomData(self)
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
            grids.extend(cgrids)
        elif self.num_times > 1:
            # Create time series for series containing multiple times as frames
            tgrids = []
            for t in range(self.num_times):
                g = DicomGrid(d, time=t)
                g.series_index = t
                tgrids.append(g)
            grids.extend(tgrids)
        else:
            # Create single channel, single time series.
            g = DicomGrid(d)
            rs = getattr(self, 'refers_to_series', None)
            if rs:
                # If this associated with another series (e.g. is a segmentation), make
                # it a channel together with that associated series.
                derived.append((g, rs))
            else:
                sgrids[self] = gg = [g]
                grids.extend(gg)
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
        #for gg in grids[1:]:
        #    for g in gg:
        #        g.show_on_open = False
        return grids

    @property
    def sort_key(self):
        return self.patient_id, self.study_date, self.name

    @property
    def plane_uids(self):
        return tuple(fi.instance_uid for fi in self.files)

    @property
    def ref_plane_uids(self):
        if len(self.files) == 1 and hasattr(self.files[0], 'ref_instance_uids'):
            uids = self.files[0].ref_instance_uids
            if uids is not None:
                return tuple(uids)
        return None

    @property
    def num_times(self):
        if self._num_times is None:
            nt = self.sample_file.get('NumberOfTemporalPositions', None)
            if nt is None:
                times = sorted(set(data.trigger_time for data in self.files))
                nt = len(times)
                for data in self.files:
                    data._time = times.index(data.trigger_time) + 1
                    data.inferred_properties += "TemporalPositionIdentifier"
                if nt > 1:
                    self.session.logger.warning(
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
        if len(self._raw_files) <= 1:
            return
        reference_file = self._raw_files[0]
        if hasattr(reference_file, "SliceLocation"):
            self._raw_files.sort(key=lambda x: x.SliceLocation)
        elif hasattr(reference_file, "ImageIndex"):
            self._raw_files.sort(key=lambda x: x.ImageIndex)

    def _validate_time_series(self):
        if self.num_times == 1:
            return

        for fi in self.files:
            if fi._time is None:
                raise ValueError('Missing dicom TemporalPositionIdentifier for image %s' % fi.path)

        tset = set(fi._time for fi in self.files)
        if len(tset) != self.num_times:
            msg = ('DICOM series header says it has %d times but %d found, %s... %d files.'
                    % (self.num_times, len(tset), self.files[0].path, len(self.files)))
            self.session.logger.warning(msg)
            self._num_times = len(tset)

        tcount = {t: 0 for t in tset}
        for fi in self.files:
            tcount[fi._time] += 1
        nz = len(self.files) / self.num_times
        for t, c in tcount.items():
            if c != nz:
                raise ValueError(
                    'DICOM time series time %d has %d images, expected %d'
                    % (t, c, nz)
                    )

    def grid_size(self):
        xsize, ysize = self.columns, self.rows
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

        return xsize, ysize, zsize

    def origin(self):
        if len(self.files) == 0:
            return None
        pos = self.files[0]._position
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

    def pixel_spacing(self):
        pspacing = self.sample_file.get('PixelSpacing')
        if pspacing is None and self.multiframe:
            pspacing = self.files[0].pixel_spacing

        if pspacing is None:
            xs = ys = 1
            self.session.logger.warning('Missing PixelSpacing, using value 1, %s' % self.paths[0])
        else:
            xs, ys = [float(s) for s in pspacing]
        zs = self.z_plane_spacing()
        if zs is None:
            nz = self.grid_size()[2]
            if nz > 1:
                self.session.logger.warning(
                    'Cannot determine z spacing, missing ImagePositionPatient, using value 1, %s'
                    % self.paths[0]
                    )
            zs = 1  # Single plane image
        elif zs == 0:
            self.session.logger.warning('Error. Image planes are at same z-position.  Setting spacing to 1.')
            zs = 1

        return (xs, ys, zs)

    def z_plane_spacing(self):
        dz = self._z_spacing
        if dz is None:
            files = self.files
            if self.multiframe:
                f = files[0]
                fpos = f.frame_positions
                if fpos is None:
                    gfov = f.grid_frame_offset_vector
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
        # Try to calculate spacing based on file spacing first
        spacings = [(z1 - z0) for z0, z1 in zip(z[:-1], z[1:])]
        dzmin, dzmax = min(spacings), max(spacings)
        tolerance = 1e-3 * max(abs(dzmax), abs(dzmin))
        dz = dzmax if abs(dzmax) > abs(dzmin) else dzmin
        if dzmax - dzmin > tolerance:
            msg = ('Plane z spacings are unequal, min = %.6g, max = %.6g, using max.\n' % (dzmin, dzmax) +
                   'Perpendicular axis (%.3f, %.3f, %.3f)\n' % tuple(self.plane_normal()) +
                   'Directory %s\n' % os.path.dirname(self.files[0].path) +
                   '\n'.join(['%s %s' % (os.path.basename(f.path), f._position) for f in self.files]))
            self.session.logger.warning(msg)
            # If we're over the threshold try to get it from SliceThickness * SliceSpacing
            thickness = self.files[0].SliceThickness or 1
            spacing = self.files[0].SliceSpacing or 1
            spacing = thickness * spacing
            dz = spacing
        return dz

    @property
    def has_image_data(self):
        return (self.bits_allocated and self.pixel_representation)

    @property
    def dicom_class(self):
        cuid = self.sop_class_uid
        return 'unknown' if cuid is None else cuid.name

    @requires_gui
    def show_info(self):
        pass


class DicomData:
    def __init__(self, series):
        self.dicom_series = series
        self.paths = tuple(series.paths)
        npaths = len(series.paths)  # noqa assigned but not accessed
        self.name = series.name
        rsi = self.dicom_series.rescale_intercept
        if rsi == int(rsi):
            rsi = int(rsi)
        self.rescale_intercept = rsi
        self.rescale_slope = self.dicom_series.rescale_slope
        bits = self.dicom_series.bits_allocated
        rep = self.dicom_series.pixel_representation
        self.value_type = self.numpy_value_type(bits, rep, self.rescale_slope, self.rescale_intercept)
        self.dicom_series.samples_per_pixel
        ns = self.dicom_series.samples_per_pixel
        if ns == 1:
            mode = 'grayscale'
        elif ns == 3:
            mode = 'RGB'
        else:
            raise ValueError('Only 1 or 3 samples per pixel supported, got %d' % ns)
        self.mode = mode
        self.channel = 0
        pi = self.dicom_series.photometric_interpretation
        if pi == 'MONOCHROME1':
            pass  # Bright to dark values.
        if pi == 'MONOCHROME2':
            pass  # Dark to bright values.
        ppv = self.dicom_series.pixel_padding
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
        # TODO: Don't need to dcmread already read in data...
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
        # TODO: Get the pixel array from the SeriesFiles
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


class SeriesFile:
    def __init__(self, data):
        self.data = data
        self.path = data.filename
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
        self.grid_frame_offset_vector = [float(o) for o in gfov] if gfov is not None else None
        self.class_uid = getattr(data, 'SOPClassUID', None)
        self.instance_uid = getattr(data, 'SOPInstanceUID', None)
        self.ref_instance_uid = getattr(data, 'ReferencedSOPInstanceUID', None)
        self.pixel_spacing = None
        self.frame_positions = None
        if self._num_frames is not None:
            def floats(s):
                return [float(x) for x in s]

            self.pixel_spacing = self._sequence_elements(
                data
                , (('SharedFunctionalGroupsSequence', 1), ('PixelMeasuresSequence', 1))
                , 'PixelSpacing'
                , floats
            )
            self.frame_positions = self._sequence_elements(
                data
                , (('PerFrameFunctionalGroupsSequence', 'all'), ('PlanePositionSequence', 1))
                , 'ImagePositionPatient'
                , floats
            )
            self.ref_instance_uids = self._sequence_elements(
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
        return getattr(self.data, 'TriggerTime', None)

    @property
    def multiframe(self):
        nf = self._num_frames
        return nf is not None and nf > 1

    def __getattr__(self, item):
        # For any field that we don't override just return the pydicom attr
        return self.data.get(item)

    def __iter__(self):
        return iter(self.data)

    def _sequence_elements(self, data, seq_names, element_name, convert=None):
        """
        seq_names:    List of (value, count) tuples that indicate the sequence to be
                      inspected and how many values to return. Can be an integer or 'all'.
        element_name: The final element name we're looking for
        convert:      Any callable to be applied to the values returned.

        Basically, recursively walk the list of sequence names in the DICOM dataset
        looking for element_name, and return it when found.
        """
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
