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

class _ReadPBondsBundle(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class PBInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from . import readpbonds
                    return readpbonds.read_pseudobond_file(session, data, file_name)
        else:
            from chimerax.save_command import SaverInfo
            class PBInfo(SaverInfo):
                def save(self, session, path, **kw):
                    from . import readpbonds
                    readpbonds.write_pseudobond_file(session, path, **kw)

                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg, BoolArg
                    return {
                        'models': ModelsArg,
                        'selected_only': BoolArg,
                    }

        return PBInfo()

bundle_api = _ReadPBondsBundle()
