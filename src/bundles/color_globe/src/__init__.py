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

class _ColorGlobeAPI(BundleAPI):

    @staticmethod
    def open_file(session, f, name, **kw):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        from . import dres
        return dres.read_directional_resolution(session, f, name, **kw)

bundle_api = _ColorGlobeAPI()
