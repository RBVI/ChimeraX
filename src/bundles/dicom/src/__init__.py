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
from chimerax.map import add_map_format
from chimerax.core.tools import get_singleton

from .dicom import (
    DICOMMapFormat, DicomOpener, fetchers,
    DICOMBrowserTool, DICOMDatabases
)


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
        # return runners['name']
        if name == "DICOM medical imaging":
            return DicomOpener()
        else:
            return fetchers[name]()


bundle_api = _DICOMBundle()
