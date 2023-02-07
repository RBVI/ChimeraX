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
__version__ = "1.0"
from .nifti import NifTI

from chimerax.core.toolshed import BundleAPI
from chimerax.open_command import OpenerInfo

class _NifTIBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        class NiftiOpenerInfo(OpenerInfo):
            def open(self, session, data, *args, **kw):
                nifti = NifTI.from_paths(session, data)
                return nifti.open()
        return NiftiOpenerInfo()

bundle_api = _NifTIBundle()
