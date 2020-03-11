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

from .sdf import read_sdf

from chimerax.core.toolshed import BundleAPI

class _SDF_API(BundleAPI):

    @staticmethod
    def open_file(session, stream, file_name):
        return read_sdf(session, stream, file_name)

    @staticmethod
    def run_provider(session, name, mgr, *, operation=None, data=None, file_name=None):
        if operation == "open args":
            return {}
        elif operation == "open":
            return read_sdf(session, data, file_name)

bundle_api = _SDF_API()
