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

from chimerax.open_command import OpenerInfo

from .dicom import register_dicom_format, open_dicom

class _DICOMBundle(BundleAPI):
    @staticmethod
    def initialize(session, bundle_info):
        """Register file formats, commands, and database fetch."""
        register_dicom_format(session)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        class DicomOpenerInfo(OpenerInfo):
            def open(self, session, data, file_name, **kw):
                return open_dicom(session, data)
        return DicomOpenerInfo()


bundle_api = _DICOMBundle()
