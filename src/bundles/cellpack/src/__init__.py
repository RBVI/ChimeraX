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
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_command import FetcherInfo
        class CellpackFetcherInfo(FetcherInfo):
            def fetch(self, session, ident, format_name, ignore_cache, **kw):
                from .fetch_cellpack import fetch_cellpack
                return fetch_cellpack(session, ident, ignore_cache=ignore_cache)
        return CellpackFetcherInfo()

bundle_api = _MyAPI()
