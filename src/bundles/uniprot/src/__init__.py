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

from .fetch_uniprot import map_uniprot_ident

from chimerax.core.toolshed import BundleAPI

class _UniprotBundleAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr, *, widget_info=None, **kw):
        from chimerax.open_command import FetcherInfo
        class UniprotFetcherInfo(FetcherInfo):
            def fetch(self, session, ident, format_name, ignore_cache, **kw):
                from .fetch_uniprot import fetch_uniprot
                return fetch_uniprot(session, ident, ignore_cache=ignore_cache)
        return UniprotFetcherInfo()

bundle_api = _UniprotBundleAPI()
