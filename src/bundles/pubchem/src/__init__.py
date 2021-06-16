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

class _PubChemAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import FetcherInfo
            class PubchemFetcherInfo(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache, *, res_name=None, **kw):
                    from . import pubchem
                    return pubchem.fetch_pubchem(session, ident, ignore_cache=ignore_cache,
                        res_name=res_name, **kw)

                @property
                def fetch_args(self):
                    from chimerax.core.commands import StringArg
                    return { 'res_name': StringArg }
            return PubchemFetcherInfo()
        else:
            from .build_ui import PubChemProvider
            return PubChemProvider(session)

bundle_api = _PubChemAPI()
