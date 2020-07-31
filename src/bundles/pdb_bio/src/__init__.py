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

class _PDBBioAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            if name in ('pdbe_bio', 'rcsb_bio'):
                if name == 'pdbe_bio':
                    site = 'pdbe'
                elif name == 'rcsb_bio':
                    site = 'rcsb'
                from chimerax.open_command import FetcherInfo
                class PDBBioFetcherInfo(FetcherInfo):
                    def fetch(self, session, id, format_name, ignore_cache, site=site, **kw):
                        from .fetch_pdb_bio import fetch_pdb_biological_assemblies
                        models, status = fetch_pdb_biological_assemblies(session, id, site=site,
                                                                         ignore_cache=ignore_cache, **kw)
                        return models, status
                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import PositiveIntArg
                        return {'max_assemblies': PositiveIntArg}
                return PDBBioFetcherInfo()

bundle_api = _PDBBioAPI()
