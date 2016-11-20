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

class _MyAPI(BundleAPI):

    @staticmethod
    def open_file(session, f, name, filespec=None, format_name=None, **kw):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        import sys
        import ihm
        return ihm.read_ihm(session, filename, name, **kw)

    @staticmethod
    def save_file(session, name, format_name=None, **kw):
        # 'save_file' is called by session code to save a file
        import sys
        import savecoords
        return savecoords.save_binary_coordinates(session, name, **kw)

bundle_api = _MyAPI()
