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


class _VTKBundleAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class VtkInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from . import vtk
                    return vtk.read_vtk(session, data, file_name)
        else:
            from chimerax.save_command import SaverInfo
            class VtkInfo(SaverInfo):
                def save(self, session, path, *, models=None, **kw):
                    from .export_vtk import write_scene_as_vtk
                    write_scene_as_vtk(session, path, models)

                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg
                    return { 'models': ModelsArg }
        return VtkInfo()

bundle_api = _VTKBundleAPI()
