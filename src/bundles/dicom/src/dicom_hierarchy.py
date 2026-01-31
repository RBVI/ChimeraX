# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import datetime
import math

from collections import defaultdict
from functools import cached_property


from numpy import cross, float32, uint8, int8, uint16, int16

from typing import Optional

from chimerax.core.decorators import requires_gui
from chimerax.core.models import Model
from chimerax.core.tools import get_singleton
from chimerax.core.session import Session

from chimerax.map.volume import open_grids
from chimerax.segmentations import open_grids_as_segmentation

from .errors import MismatchedUIDError, UnrenderableSeriesError
from .dicom_models import DicomContours, DicomGrid
from . import modality


class Patient(Model):
    """A set of DICOM files that have the same Patient ID"""

    def __init__(self, session: Session, pid: str):
        self.session = session
        self.pid = pid
        Model.__init__(self, "Patient %s" % pid, session)
        self.studies = []

    def studies_from_files(self, files) -> None:
        studies = defaultdict(list)
        for f in files:
            if hasattr(f, "StudyInstanceUID"):
                studies[f.StudyInstanceUID].append(f)
            else:
                studies["Unknown Study"].append(f)
        for key, files in studies.items():
            study = Study(self.session, key, self)
            study.series_from_files(files)
            self.studies.append(study)
        if studies:
            self.name = f"Patient (ID: {self.patient_id})"

    @requires_gui
    def show_info(self):
        from .ui import DICOMBrowserTool

        tool = get_singleton(self.session, DICOMBrowserTool, "DICOM Browser")
        tool.display(True)

    def merge_and_delete_other(self, other):
        if not other.pid == self.pid:
            raise MismatchedUIDError(
                "Can't merge patients that don't have the same Patient ID"
            )
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
            return datetime.datetime.strptime(self.birth_date, "%Y%m%d")
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
        except ValueError:  # Already in scene
            pass
        for study in self.studies:
            self.add([study])
            study.series.sort(key=lambda x: x.number)
            study.open_series_as_models()

    def __iter__(self):
        return iter(self.studies)

    def take_snapshot(self, session, flags):
        data = super().take_snapshot(session, flags)
        data["patient_id"] = self.pid
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        pid = data.get("patient_id", None)
        new_patient = cls(session, pid)
        Model.set_state_from_snapshot(new_patient, session, data)
        return new_patient


class Study(Model):
    """A set of DICOM files that have the same Study Instance UID"""

    def __init__(self, session, uid, patient: Patient):
        import pydicom.uid

        if type(uid) is str:
            uid = pydicom.uid.UID(uid)
        self.uid = uid
        self.session = session
        self.patient = patient
        Model.__init__(self, "Study (%s)" % uid, session)
        self.series = []  # regular images
        self._model_watchers = {}
        self.series_models: dict[str, list] = {}

    def series_from_files(self, files) -> None:
        series = defaultdict(list)
        for f in files:
            if hasattr(f, "SeriesInstanceUID"):
                series[f.SeriesInstanceUID].append(f)
            else:
                series["Unknown Series"].append(f)
        for key, files in series.items():
            s = Series(self.session, self, files)
            self.series.append(s)
        if self.series:
            if self.date_as_datetime:
                self.name = "%s Study (%s)" % (
                    self.body_part,
                    self.date_as_datetime.strftime("%Y-%m-%d"),
                )
            else:
                self.name = "%s Study (Unknown Date)" % (self.body_part)
            self.series.sort(key=lambda s: s.sort_key)
        plane_ids = {s.plane_uids: s for s in self.series}
        for s in self.series:
            ref = s.ref_plane_uids
            if ref and ref in plane_ids:
                s.refers_to_series = plane_ids[ref]

    @requires_gui
    def show_info(self):
        from .ui import DICOMBrowserTool

        tool = get_singleton(self.session, DICOMBrowserTool, "DICOM Browser")
        tool.display(True)

    def merge_and_delete_other(self, other):
        if not other.uid == self.uid:
            raise MismatchedUIDError(
                "Can't merge studies that don't have the same Study UID"
            )
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
            derived = []
            sgrids = {}
            all_opened_models = []
            for s in self.series:
                try:
                    if s.uid not in self.series_models:
                        models = s.to_models(all_opened_models, derived, sgrids)
                        all_opened_models.extend(models)
                        for model in models:
                            self._model_watchers[model] = model.triggers.add_handler(
                                "deleted", self._on_child_model_deleted
                            )
                        self.add(models)
                        self.series_models[s.uid] = models
                except UnrenderableSeriesError as e:
                    self.session.logger.warning(str(e))

    def _on_child_model_deleted(self, _, model):
        for s in self.series:
            try:
                smodels = self.series_models[s.uid]
                if model in smodels:
                    smodels.remove(model)
                    if not smodels:
                        del self.series_models[s.uid]
            except KeyError:
                # This can happen if we remove all models
                pass
        model.triggers.remove_handler(self._model_watchers[model])
        del self._model_watchers[model]

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
            return datetime.datetime.strptime(self.series[0].study_date, "%Y%m%d")
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

    def take_snapshot(self, session, flags):
        data = super().take_snapshot(session, flags)
        data["study_id"] = str(self.uid)
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        pid = data.get("study_id", None)
        new_study = cls(session, pid, data["parent"])
        Model.set_state_from_snapshot(new_study, session, data)
        return new_study


class Series:
    """Set of DICOM files (.dcm suffix) that have the same series unique identifier (UID)."""

    def __init__(self, session, parent, files):
        self.session = session
        self.parent_study = parent
        self.files = files
        self.sample_file = self.files[0]
        self._refers_to_series = None
        self.files_by_size = defaultdict(list)
        self.dicom_data = []
        for f in self.files:
            self.files_by_size[f.size].append(f)
        for file_list in self.files_by_size.values():
            if len(file_list) == 1:
                file = file_list[0]
                if file.is_multimask:
                    for i in range(file.number_of_masks):
                        self.dicom_data.append(
                            DicomData(self.session, self, file_list, mask_number=i)
                        )
                else:
                    self.dicom_data.append(DicomData(self.session, self, file_list))
            else:
                self.dicom_data.append(DicomData(self.session, self, file_list))

    def to_models(self, open_models, derived, sgrids):
        models = []
        for data in self.dicom_data:
            grids = data.to_models(open_models, derived, sgrids)
            models.extend(grids)
        return models

    @property
    def refers_to_series(self):
        return self._refers_to_series

    @refers_to_series.setter
    def refers_to_series(self, series):
        self._refers_to_series = series
        for data in self.dicom_data:
            data.refers_to_series = series

    @property
    def name(self):
        return f"{self.number} {self.modality} ({self.description})"

    @property
    def sop_class_uid(self):
        return self.sample_file.get("SOPClassUID")

    @property
    def pixel_padding(self):
        return self.sample_file.get("PixelPaddingValue")

    @property
    def bits_allocated(self):
        return self.sample_file.get("BitsAllocated")

    @property
    def pixel_representation(self):
        return self.sample_file.get("PixelRepresentation")

    @property
    def photometric_interpretation(self):
        return self.sample_file.get("PhotometricInterpretation")

    @property
    def rescale_slope(self):
        return self.sample_file.get("RescaleSlope", 1)

    @property
    def rescale_intercept(self):
        return self.sample_file.get("RescaleIntercept", 0)

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
        if self.sample_file.get("SeriesNumber", None) is None:
            self.session.logger.warning("SeriesNumber not specified; setting to 0")
            return 0
        else:
            return int(self.sample_file.get("SeriesNumber", 0))

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

    @property
    def sort_key(self):
        return self.patient_id, self.study_date, self.name

    @property
    def plane_uids(self):
        uids = []
        for data in self.dicom_data:
            uids.extend([fi.instance_uid for fi in data.files])
        return tuple(uids)

    @property
    def ref_plane_uids(self):
        uids = []
        for data in self.dicom_data:
            if len(data.files) == 1 and hasattr(data.sample_file, "ref_instance_uids"):
                _uids = data.files[0].ref_instance_uids
                if _uids:
                    for uid in _uids:
                        uids.append(uid)
        return tuple(uids) if uids else None

    @property
    def has_image_data(self):
        return self.bits_allocated and self.pixel_representation

    @property
    def dicom_class(self):
        cuid = self.sop_class_uid
        return "unknown" if cuid is None else cuid.name

    @requires_gui
    def show_info(self):
        pass


class DicomData:
    def __init__(
        self,
        session,
        series,
        files: list["SeriesFile"],
        mask_number: Optional[int] = None,
    ):
        import pydicom.uid

        self.session = session
        self.mask_number = mask_number
        self.dicom_series = series
        self.files = files
        self.order_slices()
        self.sample_file = files[0]
        self.paths = tuple(file.path for file in files)
        self._refers_to_series = None
        self.transfer_syntax = None
        self._multiframe = None
        self._num_times = None
        self.image_series = True
        self.contour_series = False
        if any([f.SOPClassUID == pydicom.uid.RTStructureSetStorage for f in files]):
            self.image_series = False
            self.contour_series = True
        if not any([f.get("PixelData") for f in files]):
            self.image_series = False
        if self.transfer_syntax is None and hasattr(
            self.sample_file.file_meta, "TransferSyntaxUID"
        ):
            self.transfer_syntax = self.sample_file.file_meta.TransferSyntaxUID
        # if len(files) == 1 and self.multiframe:
        #    # Determination of whether frames reversed done in z_plane_spacing()
        #    self.z_plane_spacing()
        # Check that time series images all have time value, and all times are found
        self._validate_time_series()
        npaths = len(self.paths)  # noqa assigned but not accessed
        self.name = series.name
        if self.mask_number is not None:
            self.name += " (Mask %d)" % (self.mask_number + 1)
        rsi = self.dicom_series.rescale_intercept
        if rsi == int(rsi):
            rsi = int(rsi)
        self.rescale_intercept = rsi
        self.rescale_slope = int(self.dicom_series.rescale_slope)
        if self.image_series and not self.contour_series:
            bits = self.sample_file.get("BitsAllocated")
            rep = self.sample_file.get("PixelRepresentation")
            self.value_type = self.numpy_value_type(
                bits, rep, self.rescale_slope, self.rescale_intercept
            )
            ns = self.samples_per_pixel
            if ns == 1:
                mode = "grayscale"
            elif ns == 3:
                mode = "RGB"
            else:
                raise ValueError("Only 1 or 3 samples per pixel supported, got %d" % ns)
            self.mode = mode
        self.channel = 0
        pi = self.dicom_series.photometric_interpretation
        if pi == "MONOCHROME1":
            pass  # Bright to dark values.
        if pi == "MONOCHROME2":
            pass  # Dark to bright values.
        ppv = self.dicom_series.pixel_padding
        if ppv is not None:
            self.pad_value = self.rescale_slope * ppv + self.rescale_intercept
        else:
            self.pad_value = None
        self.files_are_3d = self.multiframe
        self._reverse_planes = False  # (self.multiframe and self._reverse_frames)
        self.data_size = self.grid_size()
        self.data_step = self.pixel_spacing()
        self.data_origin = origin = self.origin()
        if origin is None:
            self.origin_specified = False
            self.data_origin = (0, 0, 0)
        else:
            self.origin_specified = True
        self.data_rotation = self.rotation()
        self.transfer_syntax = None
        self._multiframe = None
        self._reverse_frames = False
        self._num_times = None
        self._z_spacing = None
        # if len(files) == 1 and self.multiframe:
        #    # Determination of whether frames reversed done in z_plane_spacing()
        #    self.z_plane_spacing()
        # Check that time series images all have time value, and all times are found
        self._validate_time_series()

    def _to_grids(self, derived, sgrids) -> list["DicomGrid"]:
        grids = []
        # derived = []  # For grouping derived series with original series
        # sgrids = {}
        if self.mode == "RGB":
            # Create 3-channels for RGB series
            cgrids = []
            colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
            suffixes = [" red", " green", " blue"]
            for channel in (0, 1, 2):
                g = DicomGrid.from_series(self, channel=channel)
                if self.dicom_series.modality == "SEG":
                    g.initial_plane_display = False
                g.name += suffixes[channel]
                g.rgba = colors[channel]
                cgrids.append(g)
            grids.extend(cgrids)
        elif self.num_times > 1:
            # Create time series for series containing multiple times as frames
            tgrids = []
            for t in range(self.num_times):
                g = DicomGrid.from_series(self, time=t)
                if self.dicom_series.modality == "SEG":
                    g.initial_plane_display = False
                g.series_index = t
                tgrids.append(g)
            grids.extend(tgrids)
        else:
            # Create single channel, single time series.
            g = DicomGrid.from_series(self)
            if self.dicom_series.modality == "SEG":
                g.initial_plane_display = False
            rs = getattr(self, "refers_to_series", None)
            if rs:
                # If this associated with another series (e.g. is a segmentation), make
                # it a channel together with that associated series.
                # the bool is whether it's been opened
                derived.append([g, rs, False])
            else:
                sgrids[self.dicom_series] = gg = [g]
                grids.extend(gg)
        # Group derived series with the original series
        channel_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        for index, (g, rs, _) in enumerate(derived):
            # open_series = set([s.dicom_series for s in sgrids.keys()])
            if rs in sgrids:
                source_grid = sgrids[rs]
                # if len(sg) == 1:
                #    sg[0].channel = 1
                # sg.append(g)
                # g.channel = len(sg)
                # g.rgba = channel_colors[(g.channel - 2) % len(channel_colors)]
                # if not g.dicom_data.origin_specified:
                #    # Segmentation may not have specified an origin
                #    g.set_origin(sg[0].origin)
                g.reference_data = source_grid[0]
            if not derived[index][2]:
                grids.append(g)
                derived[index][2] = True
        # Show only first group of grids
        # for gg in grids[1:]:
        #    for g in gg:
        #        g.show_on_open = False
        return grids

    def to_models(self, open_models, derived, sgrids):
        if self.contour_series:
            return [DicomContours(self.session, s, self.name) for s in self.files]
        elif self.image_series:
            grids = self._to_grids(derived, sgrids)
            originals, segs = [], []
            for grid in grids:
                if grid.dicom_data.modality == modality.Segmentation:
                    segs.append(grid)
                else:
                    originals.append(grid)
            models = []
            # TODO: Look thorugh open_models for segmentations' source volumes
            if originals:
                models.extend(open_grids(self.session, originals, name=self.name)[0])
            if segs:
                models.extend(
                    open_grids_as_segmentation(self.session, segs, name=self.name)[0]
                )
            for model in models:
                for omodel in open_models:
                    if getattr(model.data, "reference_data", None) is omodel.data:
                        model.reference_data = omodel
            return models
        else:
            raise UnrenderableSeriesError(
                "No model created for Series #%s from patient %s because "
                "it had no pixel data. Metadata will still "
                "be available." % (self.number, self.patient_id)
            )

    @property
    def number(self):
        if self.sample_file.get("SeriesNumber", None) is None:
            self.session.logger.warning("SeriesNumber not specified; setting to 0")
            return 0
        else:
            return int(self.sample_file.get("SeriesNumber", 0))

    @property
    def patient_id(self):
        return self.sample_file.get("PatientID", "")

    @property
    def columns(self):
        return self.sample_file.get("Columns")

    @property
    def rows(self):
        return self.sample_file.get("Rows")

    @property
    def samples_per_pixel(self):
        return self.sample_file.get("SamplesPerPixel")

    def _validate_time_series(self):
        if self.num_times == 1:
            return

        for fi in self.files:
            if fi._time is None:
                raise ValueError(
                    "Missing dicom TemporalPositionIdentifier for image %s" % fi.path
                )

        tset = set(fi._time for fi in self.files)
        if len(tset) != self.num_times:
            msg = (
                "DICOM series header says it has %d times but %d found, %s... %d files."
                % (self.num_times, len(tset), self.files[0].path, len(self.files))
            )
            self.session.logger.warning(msg)
            self._num_times = len(tset)

        tcount = {t: 0 for t in tset}
        for fi in self.files:
            tcount[fi._time] += 1
        nz = len(self.files) / self.num_times
        for t, c in tcount.items():
            if c != nz:
                raise ValueError(
                    "DICOM time series time %d has %d images, expected %d" % (t, c, nz)
                )

    @property
    def num_times(self):
        if self._num_times is None:
            nt = self.sample_file.get("NumberOfTemporalPositions", None)
            if nt is None:
                times = sorted(set(data.trigger_time for data in self.files))
                nt = len(times)
                for data in self.files:
                    data._time = times.index(data.trigger_time) + 1
                    data.inferred_properties.add("TemporalPositionIdentifier")
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
        # TODO: Double check if need be? Order these orderings by reliability?
        if len(self.files) <= 1:
            return
        reference_file = self.files[0]
        if hasattr(reference_file, "SliceLocation") and reference_file.get(
            "SliceLocation", None
        ):
            self.files.sort(key=lambda x: (x.get("TriggerTime", 1), x.SliceLocation))
        elif hasattr(reference_file, "ImageIndex") and reference_file.get(
            "ImageIndex", None
        ):
            self.files.sort(key=lambda x: x.ImageIndex)
        elif hasattr(reference_file, "AcquisitionNumber") and reference_file.get(
            "AcquisitionNumber", None
        ):
            self.files.sort(key=lambda x: x.AcquisitionNumber)
        else:
            self.files.sort(key=lambda x: x.position[2])

    def grid_size(self):
        xsize, ysize = self.columns, self.rows
        files = self.files
        if self.multiframe:
            if len(files) == 1:
                if self.mask_number is not None:
                    zsize = self.files[0].mask_length
                else:
                    zsize = self.files[0]._num_frames
            else:
                maxf = max(fi._num_frames for fi in files)
                raise ValueError(
                    "DICOM multiple paths (%d), with multiple frames (%d) not supported, %s"
                    % (len(self.paths), maxf, files[0].path)
                )  # noqa npaths not defined
        else:
            if self.mask_number is not None:
                zsize = self.files[0].mask_length
            else:
                zsize = len(files) // self.num_times
        return xsize, ysize, zsize

    @cached_property
    def affine(self):
        files = self.files
        # According to https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-affine-formula
        # the DICOM 3D affine is computed as follows:
        # ImageOrientationPatient gives the first two (X and Y) columns. The Z-column is computed
        # either from taking the difference between each element of the ImagePositionPatient of the
        # first and last slices divided by the number of files in the dataset or by the cross
        # product of the X and Y vectors.
        orient = files[0]._orientation or [1, 0, 0, 0, 1, 0]
        position = files[0].position
        x_space, y_space = files[0].pixel_spacing
        # z_space = self._new_z_plane_spacing()
        z_space = 1
        x_axis, y_axis = orient[0:3], orient[3:6]
        z_axis = [0, 0, 1]
        if orient == (1, 0, 0, 0, 0, -1):
            y_axis = [0, 1, 0]
        if len(self.files) > 1:
            z_axis = self._z_spacing_from_files(
                files[0].position,
                files[-1].position,
                int(len(self.files) / self.num_times),
            )
            if not z_axis:
                z_axis = cross(x_axis, y_axis)
        else:
            if hasattr(self.files[0], "frame_postions") and bool(
                getattr(self.files[0], "frame_positions", [])
            ):
                if len(self.files) == 1:
                    if self.files[0].is_multimask:
                        z_axis = self._z_spacing_from_files(
                            files[0].frame_positions[0],
                            files[0].frame_positions[-1],
                            self.files[0].mask_length,
                        )
                    else:
                        z_axis = self._z_spacing_from_files(
                            files[0].frame_positions[0],
                            files[0].frame_positions[-1],
                            int(len(files[0].frame_positions) / self.num_times),
                        )
                else:
                    z_axis = self._z_spacing_from_files(
                        files[0].frame_positions[0],
                        files[0].frame_positions[-1],
                        int(len(files[0].frame_positions) / self.num_times),
                    )
                if not z_axis:
                    z_axis = cross(x_axis, y_axis)
            else:
                z_axis = cross(x_axis, y_axis)
        affine = [
            [
                x_space * x_axis[0],
                y_space * y_axis[0],
                z_space * z_axis[0],
                position[0],
            ],
            [
                x_space * x_axis[1],
                y_space * y_axis[1],
                z_space * z_axis[1],
                position[1],
            ],
            [
                x_space * x_axis[2],
                y_space * y_axis[2],
                z_space * z_axis[2],
                position[2],
            ],
            [0, 0, 0, 1],
        ]
        return affine

    def _z_spacing_from_files(self, first_pos, last_pos, num_files):
        if first_pos == last_pos and first_pos == [0, 0, 0]:
            return None
        z_axis = [
            (last_pos[0] - first_pos[0]) / num_files,
            (last_pos[1] - first_pos[1]) / num_files,
            (last_pos[2] - first_pos[2]) / num_files,
        ]
        if all(x == 0 for x in z_axis):
            return None
        if z_axis[2] == 0:
            z_axis[2] = 1
        return z_axis

    @property
    def inferior_to_superior(self):
        if len(self.files) == 1:
            return False
        # neg1 < neg2 < 0
        if self.files[0].position[2] > self.files[1].position[2]:
            return False
        # 0 < pos1 < pos 2
        else:
            return True

    def pixel_spacing(self):
        affine = self.affine
        x_vector = [affine[0][0], affine[1][0], affine[2][0]]
        y_vector = [affine[0][1], affine[1][1], affine[2][1]]
        z_vector = [affine[0][2], affine[1][2], affine[2][2]]
        x_scale = math.sqrt(sum([i**2 for i in x_vector]))
        y_scale = math.sqrt(sum([i**2 for i in y_vector]))
        z_scale = math.sqrt(sum([i**2 for i in z_vector]))
        return x_scale, y_scale, z_scale

    def rotation(self):
        # affine = self.affine
        # x_scale, y_scale, z_scale = self.pixel_spacing()
        # rotation_matrix = [
        #    [affine[0][0] / x_scale, affine[0][1] / y_scale, affine[0][2] / z_scale],
        #    [affine[1][0] / x_scale, affine[1][1] / y_scale, affine[1][2] / z_scale],
        #    [affine[2][0] / x_scale, affine[2][1] / y_scale, affine[2][2] / z_scale],
        # ]
        # We're ignoring the rotation given by the DICOM files until someone complains about it.
        # Doing this simplifies other areas of the codebase significantly.
        # 1) The plane viewers use orthographic cameras pointed down the X, Y, and Z axes, and
        #    ignoring the rotations of the files means we don't have to calculate new axes to
        #    point the cameras down when files aren't axis aligned.
        # 2) We don't have to modify the raycasting shader to do such calculations either.
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # return rotation_matrix

    def origin(self):
        affine = self.affine
        return [affine[0][3], affine[1][3], affine[2][3]]

    def read_matrix(
        self, ijk_origin, ijk_size, ijk_step, time, channel, array, progress
    ):
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
            array[:] = a[
                k0 : k0 + ksz : kstep, j0 : j0 + jsz : jstep, i0 : i0 + isz : istep
            ]
        else:
            for k in range(k0, k0 + ksz, kstep):
                if progress:
                    progress.plane((k - k0) // kstep)
                p = self.read_plane(k, time, channel, rescale=False)
                array[(k - k0) // kstep, :, :] = p[
                    j0 : j0 + jsz : jstep, i0 : i0 + isz : istep
                ]
        if self.rescale_slope != 1:
            array *= self.rescale_slope
        if self.rescale_intercept != 0:
            array += self.rescale_intercept
        return array

    def read_plane(self, k, time=None, channel=None, rescale=True):
        # TODO: Don't need to dcmread already read in data...
        from pydicom import dcmread

        if self._reverse_planes:
            klast = self.data_size[2] - 1
            k = klast - k
        if self.files_are_3d:
            d = dcmread(self.paths[0])
            data = d.pixel_array[k]
        else:
            p = k if time is None else (k + (self.data_size[2] * time))
            d = self.files[p]
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
        d = self.files[0]
        data = d.pixel_array
        if self.mask_number is not None:
            size_of_masks = self.files[0].mask_length
            data = data[
                self.mask_number * size_of_masks : (self.mask_number + 1)
                * size_of_masks,
                :,
                :,
            ]
        if channel is not None:
            data = data[:, :, :, channel]
        return data

    def numpy_value_type(
        self, bits_allocated, pixel_representation, rescale_slope, rescale_intercept
    ):
        # PixelRepresentation 0 = unsigned, 1 = signed
        if (
            rescale_slope != 1
            or int(rescale_intercept) != rescale_intercept
            or rescale_intercept < 0
            and pixel_representation == 0
        ):  # unsigned with negative offset
            return float32

        types = {
            (1, 0): uint8,
            (1, 1): int8,
            (8, 0): uint8,
            (8, 1): int8,
            (16, 0): uint16,
            (16, 1): int16,
        }
        if (bits_allocated, pixel_representation) in types:
            return types[(bits_allocated, pixel_representation)]
        raise ValueError("Unsupported value type, bits_allocated = %d" % bits_allocated)

    @property
    def modality(self):
        return self.dicom_series.modality


class SeriesFile:
    def __init__(self, data):
        self.data = data
        self.path = data.filename
        self.inferred_properties = set()
        orient = getattr(
            data, "ImageOrientationPatient", None
        )  # horz and vertical image axes
        self._orientation = tuple(float(p) for p in orient) if orient else None
        num = getattr(data, "InstanceNumber", None)
        self._num = int(num) if num else None
        # TODO: Should this just be order and not time?
        t = getattr(data, "TemporalPositionIdentifier", None)
        self._time = int(t) if t else None
        nf = getattr(data, "NumberOfFrames", None)
        self._num_frames = int(nf) if nf is not None else None
        gfov = getattr(data, "GridFrameOffsetVector", None)
        self.grid_frame_offset_vector = (
            [float(o) for o in gfov] if gfov is not None else None
        )
        self.class_uid = getattr(data, "SOPClassUID", None)
        self.instance_uid = getattr(data, "SOPInstanceUID", None)
        self.ref_instance_uid = getattr(data, "ReferencedSOPInstanceUID", None)
        self.frame_positions = None
        if self._num_frames is not None:
            self.frame_positions = self._sequence_elements(
                data,
                (
                    ("PerFrameFunctionalGroupsSequence", "all"),
                    ("PlanePositionSequence", 1),
                ),
                "ImagePositionPatient",
                lambda x: [float(y) for y in x],
            )
            self.ref_instance_uids = self._sequence_elements(
                data,
                (
                    ("SharedFunctionalGroupsSequence", 1),
                    ("DerivationImageSequence", 1),
                    ("SourceImageSequence", "all"),
                ),
                "ReferencedSOPInstanceUID",
            )

    def __lt__(self, im):
        if self._time == im._time:
            # Use z position instead of image number to assure right-handed coordinates.
            return self.position[2] < im.position[2]
        else:
            return self._time < im._time

    @property
    def is_multimask(self):
        if self.modality != "SEG":
            return False
        if self._num_frames == 1:
            return False
        if hasattr(self.data, "ReferencedSeriesSequence"):
            if (
                len(self.data.ReferencedSeriesSequence[0].ReferencedInstanceSequence)
                == 1
            ):
                return False
            if (
                len(self.data.ReferencedSeriesSequence[0].ReferencedInstanceSequence)
                < self._num_frames
            ):
                return True
        return False

    @property
    def mask_length(self):
        return len(self.data.ReferencedSeriesSequence[0].ReferencedInstanceSequence)

    @property
    def number_of_masks(self):
        return self._num_frames // self.mask_length

    @property
    def pixel_spacing(self):
        if self._num_frames is not None:
            if x := self._sequence_elements(
                self.data,
                (("SharedFunctionalGroupsSequence", 1), ("PixelMeasuresSequence", 1)),
                "PixelSpacing",
                lambda x: [float(y) for y in x],
            ):
                return x
            return 1, 1
        if x := self.data.get("PixelSpacing", None):
            return x
        if y := self.data.get("ImagerPixelSpacing"):
            return y
        return 1, 1

    @property
    def columns(self):
        return self.data.get("Columns")

    @property
    def rows(self):
        return self.data.get("Rows")

    @property
    def size(self):
        return self.columns, self.rows

    @property
    def position(self):
        # TODO: For some reason this breaks rendering the 4D Lung dataset?
        # Each frame in the set has a different ImagePositionPatient
        # So maybe we take this and move it to somewhere with more context
        pos = self.data.get("ImagePositionPatient", None)
        if self._num_frames is not None and pos is None:
            if self.frame_positions is not None:
                pos_x, pos_y = self.frame_positions[0][:2]
                z_origin = min(x[2] for x in self.frame_positions)
                pos = [pos_x, pos_y, z_origin]
        return tuple(float(p) for p in pos) if pos else (0, 0, 0)

    @property
    def trigger_time(self):
        return getattr(self.data, "TriggerTime", None)

    @property
    def slice_location(self):
        return self.data.get("SliceLocation", None)

    @property
    def multiframe(self):
        nf = self._num_frames
        return nf is not None and nf > 1

    @property
    def modality(self):
        return self.data.get("Modality", None)

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
            if count == "all":
                values = [
                    self._sequence_elements(e, seq_names[1:], element_name, convert)
                    for e in seq
                ]
            else:
                values = self._sequence_elements(
                    seq[0], seq_names[1:], element_name, convert
                )
            return values
