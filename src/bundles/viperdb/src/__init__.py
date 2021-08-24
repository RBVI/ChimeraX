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


class _ViperDBBundleAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class ViperDBInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from chimerax.pdb import open_pdb
                    models, info = open_pdb(session, data, file_name, **kw)
                    from chimerax.std_commands.sym import sym
                    from chimerax.atomic.args import Symmetry
                    sym(session, models, Symmetry('i222', session))
                    return models, info
            return ViperDBInfo()

bundle_api = _ViperDBBundleAPI()
