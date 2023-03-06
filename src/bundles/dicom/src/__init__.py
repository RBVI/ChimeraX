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
__version__ = "1.1"
from chimerax.core.toolshed import BundleAPI
from chimerax.core.commands import register
from chimerax.map import add_map_format
from chimerax.open_command import OpenerInfo, FetcherInfo
from chimerax.core.tools import get_singleton

from .dicom import DICOM, DICOMMapFormat
from .dicom_fetch import fetch_nbia_images
from .ui import DICOMBrowserTool, DICOMMetadata, DICOMDatabases

class _DICOMBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, bundle_info):
        """Register file formats, commands, and database fetch."""
        add_map_format(session, DICOMMapFormat())

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "DICOM Browser":
            return get_singleton(session, DICOMBrowserTool, "DICOM Browser")
        else:
            return DICOMDatabases(session)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if name == "DICOM medical imaging":
            class DicomOpenerInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    dcm = DICOM.from_paths(session, data)
                    return dcm.open()
            return DicomOpenerInfo()
        else:
            # Borrow from PDB to leave open the possibility of other DICOM databases
            fetcher = {
                'tcia': fetch_nbia_images
            }[name]
            class Info(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache, fetcher=fetcher, **kw):
                    return fetcher(session, ident, ignore_cache=ignore_cache, **kw)

                @property
                def fetch_args(self):
                    return {}
                    #from chimerax.core.commands import BoolArg, IntArg, FloatArg
            return Info()

bundle_api = _DICOMBundle()
