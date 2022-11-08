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

import os
from pydicom import dcmread
from typing import Any, Dict, TypeVar, Union

from chimerax.map_data import MapFileFormat

from .dicom_hierarchy import Patient

Path = TypeVar("Path", os.PathLike, str, bytes, None)

try:
    # We are inside GUI ChimeraX
    from chimerax.ui.gui import UI
except (ModuleNotFoundError, ImportError):
    # We could be in NoGUI ChimeraX
    try:
        from chimerax.core.nogui import UI
    except (ModuleNotFoundError, ImportError):
        pass
finally:
    try:
        _logger = UI.instance().session.logger
        _session = UI.instance().session
        from .ui import DICOMBrowserTool, DICOMMetadata
        # __all__ += ["DICOMBrowserTool", "DICOMMetadata"]
    except (NameError, AttributeError):
        # We didn't have either of ChimeraX's UIs, or they were uninitialized.
        # We're either in some other application or being used as a library.
        # Default to passed in sessions and the Python logging module
        import logging
        _session = None
        _logger = logging.getLogger()
        _logger.status = _logger.info


class DICOM:
    # TODO: Make a singleton
    def __init__(self, data: Union[Path, list[Path]], *, session = None):
        self.patients_by_id = defaultdict(list)
        self.patients = {}
        self.session = session or _session
        if type(data) is not list:
            self.paths = [data]
        else:
            self.paths = data
        self.find_dicom_series(self.paths)
        self.merge_patients_by_id()

    @classmethod
    def from_paths(cls, session, path: Union[Path, list[Path]]):
        return cls(path, session = session)

    def open(self):
        for patient in list(self.patients.values()):
            patient.render()
        # In order to present the correct DICOM hierarchy, color included, we need to
        # take control of rendering ourselves.
        return [], ""

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
        nseries = len(dfiles)
        nfiles = sum(len(dpaths) for dpaths in list(dfiles.values()))
        nsfiles = 0
        for dpaths in list(dfiles.values()):
            nsfiles += len(dpaths)
            _logger.status('Reading DICOM series %d of %d files in %d series' % (nsfiles, nfiles, nseries))
            patients = self.dicom_patients(dpaths)
            for patient in patients:
                self.patients_by_id[patient.pid].append(patient)

    def dicom_patients(self, paths) -> list['Patient']:
        """Group DICOM files into series"""
        series = defaultdict(list)
        patients = []
        for path in paths:
            d = dcmread(path)
            if hasattr(d, 'PatientID'):
                series[d.PatientID].append(d)
            else:
                series["Unknown Patient"].append(d)
        for key, series in list(series.items()):
            patient = Patient(self.session, key)
            patient.studies_from_files(series)
            patients.append(patient)
        return patients

    def files_by_directory(
        self, paths, search_directories=True, search_subdirectories=True,
        suffix='.dcm', _dfiles=None
    ) -> Dict[str, list[str]]:
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

    def merge_patients_by_id(self):
        """Iterate over the patients dictionary and merge all that have the same pid"""
        for patient_list in list(self.patients_by_id.values()):
            ref_patient = patient_list[0]
            for patient in patient_list[1:]:
                ref_patient.merge_and_delete_other(patient)
            self.patients[ref_patient.pid] = ref_patient
            del patient_list

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
