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

class _AlphaFoldBundle(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from .fetch_alphafold import fetch_alphafold
            from chimerax.open_command import FetcherInfo
            class Info(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache,
                          fetcher=fetch_alphafold, **kw):
                    return fetcher(session, ident, ignore_cache=ignore_cache, **kw)
                @property
                def fetch_args(self):
                    from chimerax.core.commands import BoolArg
                    return {
                        'color_confidence': BoolArg,
                        'trim': BoolArg,
                    }
            return Info()

bundle_api = _AlphaFoldBundle()
