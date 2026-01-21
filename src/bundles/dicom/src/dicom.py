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
import os
import warnings

from collections import defaultdict
from typing import Any, Dict, TypeVar, Union


from chimerax.core.session import Session
from chimerax.map_data import MapFileFormat

from .dicom_hierarchy import Patient, SeriesFile

Path = TypeVar("Path", os.PathLike, str, bytes, None)


class DICOM:
    # TODO: Make a singleton
    def __init__(self, data: Union[Path, list[Path]], *, session: Session):
        self.patients_by_id = defaultdict(list)
        self.patients = {}
        self.session = session
        if type(data) is not list:
            self.paths = [data]
        else:
            self.paths = data
        self.find_dicom_files(self.paths)
        self.merge_patients_by_id()

    @classmethod
    def from_paths(cls, session, path: Union[Path, list[Path]]):
        return cls(path, session=session)

    def open(self):
        for patient in list(self.patients.values()):
            patient.render()
        # In order to present the correct DICOM hierarchy, color included, we need to
        # take control of rendering ourselves.
        return [], ""

    def dicom_grids(self, paths, log=None) -> list[Any]:
        # A special opener that gets called by the session restore code
        self.patients_by_id = defaultdict(list)
        self.patients = {}
        if log:
            self.session = log.session
        self.paths = paths
        self.find_dicom_files(self.paths)
        self.merge_patients_by_id()
        dicom_grids = []
        derived = []
        sgrids = {}
        for patient in self.patients.values():
            for study in patient:
                for series in study:
                    for dicom_data in series.dicom_data:
                        if dicom_data.image_series:
                            dicom_grids.append(dicom_data._to_grids(derived, sgrids))
        return dicom_grids

    def find_dicom_files(
        self,
        paths,
        search_directories: bool = True,
        search_subdirectories: bool = True,
    ) -> None:
        """Look through directories to find dicom files and group the ones
        that belong to the same study and image series.  Also determine the order
        of the 2D images (one per file) in the 3D stack.  A series must be in a single
        directory.  If the same study and series is found in two directories, they
        are treated as two different series.
        """
        from pydicom import dcmread
        dfiles = []
        for path in paths:
            if os.path.isfile(path):
                dfiles.append(SeriesFile(dcmread(path)))
            elif os.path.isdir(path):
                dfiles.extend(self._find_dicom_files_in_directory_recursively(path))
        dfiles = self.filter_unreadable(dfiles)
        patients = self.dicom_patients(dfiles)
        for patient in patients:
            self.patients_by_id[patient.pid].append(patient)

    def filter_unreadable(self, files):
        import pydicom.uid
        try:
            import gdcm  # noqa import used elsewhere
        except ModuleNotFoundError:
            _has_gdcm = False
        else:
            _has_gdcm = True

        if _has_gdcm:
            return files  # PyDicom will use gdcm to read 16-bit lossless jpeg

        # Python Image Library cannot read 16-bit lossless jpeg.
        keep = []
        for f in files:
            if (
                f.file_meta.TransferSyntaxUID == pydicom.uid.JPEGLosslessSV1
                and f.get("BitsAllocated") == 16
            ):
                warning = (
                    "Could not read DICOM %s because Python Image Library cannot read 16-bit lossless jpeg "
                    "images. This functionality can be enabled by installing python-gdcm"
                )
                self.session.logger.warning(warning % f.filename)
            else:
                keep.append(f)
        return keep

    def _find_dicom_files_in_directory_recursively(self, path):
        from pydicom import dcmread
        from pydicom.errors import InvalidDicomError
        dfiles = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f in (".DS_Store", "Thumbs.db", "desktop.ini", "LICENSE") or any(
                    f.startswith(s) for s in ["._"]
                ):
                    continue
                try:
                    dfiles.append(SeriesFile(dcmread(os.path.join(root, f))))
                except InvalidDicomError:
                    self.session.logger.info(
                        "Pydicom could not read invalid or non-DICOM file %s; skipping."
                        % f
                    )
        return dfiles

    def dicom_patients(self, files) -> list["Patient"]:
        """Group DICOM files into series"""
        series = defaultdict(list)
        patients = []
        for f in files:
            if hasattr(f, "PatientID"):
                series[f.PatientID].append(f)
            else:
                series["Unknown Patient"].append(f)
        for key, series in list(series.items()):
            patient = Patient(self.session, key)
            patient.studies_from_files(series)
            patients.append(patient)
        return patients

    def merge_patients_by_id(self):
        """Iterate over the patients dictionary and merge all that have the same pid"""
        for patient_list in list(self.patients_by_id.values()):
            ref_patient = self._find_existing_patient(patient_list[0])
            starting_index = 0
            if not ref_patient:
                ref_patient = patient_list[0]
                starting_index = 1
            else:
                self.session.logger.warning(
                    "Merged incoming unique studies with existing patient with same ID"
                )
            for patient in patient_list[starting_index:]:
                ref_patient.merge_and_delete_other(patient)
            self.patients[ref_patient.pid] = ref_patient
            del patient_list

    def _find_existing_patient(self, patient):
        for model in self.session.models:
            if type(model) is Patient and model.pid == patient.pid:
                return model
        return None

    def __iter__(self):
        return iter(list(self.patients.values()))

    def __contains__(self, patient: str) -> bool:
        for patient in self.patients:
            if patient.pid == patient:
                return True
        return False


class DICOMMapFormat(MapFileFormat, DICOM):
    def __init__(self):
        MapFileFormat.__init__(
            self,
            "DICOM image",
            "dicom",
            ["dicom"],
            ["dcm"],
            batch=True,
            allow_directory=True,
        )

    @property
    def open_func(self):
        return self.open_dicom_grids

    def open_dicom_grids(self, paths, log):
        if isinstance(paths, str):
            paths = [paths]
        grids = self.dicom_grids(paths, log)
        return grids
