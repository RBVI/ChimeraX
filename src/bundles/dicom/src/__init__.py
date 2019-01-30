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

from chimerax.core.toolshed import BundleAPI

class _DICOMBundle(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        """Register file formats, commands, and database fetch."""
        from .dicom import register_dicom_format
        register_dicom_format(session)

    '''
    Not using this since bundle_info.xml DataFormat classifier cannot handle allow_directory = true.
    @staticmethod
    def open_file(session, stream, file_name):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        from . import dicom
        models, msg = dicom.open_dicom(session, stream, file_name)
        return models, msg
    '''

bundle_api = _DICOMBundle()
