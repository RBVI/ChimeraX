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

from .rot_lib import RotamerLibrary, RotamerParams, UnsupportedResTypeError, NoResidueRotamersError
from .manager import NoRotamerLibraryError

from chimerax.core.toolshed import BundleAPI

class _RotLibMgrBundleAPI(BundleAPI):

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        """Initialize rotamer library manager"""
        if name == "rotamers":
            from .manager import RotamerLibManager
            session.rotamers = RotamerLibManager(session, name)
        else:
            raise AssertionError("This bundle does not provide a '%s' manager" % name)

bundle_api = _RotLibMgrBundleAPI()
