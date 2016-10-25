# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    @staticmethod
    def open_file(session, f, name, filespec=None, **kw):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        from . import vtk
        return vtk.read_vtk(session, f, name, **kw)

bundle_api = _MyAPI()
