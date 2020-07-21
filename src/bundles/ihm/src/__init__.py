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

class _IHMAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr):
        if name == "Integrative/Hybrid Models":
            from chimerax.open_command import OpenerInfo
            class Info(OpenerInfo):
                def open(self, session, data, file_name, ensembles=False, **kw):
                    from . import ihm
                    return ihm.read_ihm(session, data, file_name, load_ensembles=ensembles, **kw)

                @property
                def open_args(self):
                    from chimerax.core.commands import BoolArg
                    return { 'ensembles': BoolArg }
        else:
            from chimerax.open_command import FetcherInfo
            class Info(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache, **kw):
                    from .fetch_ihm import fetch_ihm
                    return fetch_ihm(session, ident, ignore_cache, **kw)
        return Info()

bundle_api = _IHMAPI()
