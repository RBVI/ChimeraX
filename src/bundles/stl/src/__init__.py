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

class _STLAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'STLModel':
            from . import stl
            return stl.STLModel
        return None

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class STLInfo(OpenerInfo):
                def open(self, session, path, file_name, **kw):
                    from . import stl
                    return stl.read_stl(session, path, file_name)
        else:
            from chimerax.save_command import SaverInfo
            class STLInfo(SaverInfo):
                def save(self, session, path, models=None):
                    from . import stl
                    stl.write_stl(session, path, models)

                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg
                    return { 'models': ModelsArg }

        return STLInfo()
bundle_api = _STLAPI()
