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
__version__ = "1.2.7"
from chimerax.core.toolshed import BundleAPI
from chimerax.map import add_map_format
from chimerax.core.tools import get_singleton
from chimerax.dicom.dicom import DICOMMapFormat
from chimerax.dicom.dicom_volumes import DICOMVolume
from chimerax.dicom.dicom_opener import DicomOpener
from chimerax.dicom.dicom_saver import DicomSaver
from chimerax.dicom.dicom_fetcher import fetchers
from chimerax.dicom.dicom_hierarchy import Patient, Study
from chimerax.dicom.dicom_models import DicomGrid


class _DICOMBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, _):
        """Register file formats, commands, and database fetch."""
        add_map_format(session, DICOMMapFormat())

    @staticmethod
    def get_class(class_name):
        class_names = {"Patient": Patient, "Study": Study, "DICOMVolume": DICOMVolume}
        return class_names.get(class_name, None)

    @staticmethod
    def start_tool(session, _, ti):
        if ti.name == "DICOM Browser":
            from chimerax.dicom.ui import DICOMBrowserTool

            return get_singleton(session, DICOMBrowserTool, "DICOM Browser")
        else:
            from chimerax.dicom.ui import DICOMDatabases

            return get_singleton(session, DICOMDatabases, "DICOM Browser")

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        # return runners['name']
        if mgr == session.open_command:
            if name == "DICOM medical imaging":
                return DicomOpener()
            else:
                return fetchers[name]()
        elif mgr == session.save_command:
            if name == "DICOM medical imaging":
                return DicomSaver()


bundle_api = _DICOMBundle()
