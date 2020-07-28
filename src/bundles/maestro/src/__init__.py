# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_command import OpenerInfo
        class MaestroOpenerInfo(OpenerInfo):
            def open(self, session, path, file_name, **kw):
                from .io import open_mae
                return open_mae(session, path, file_name, True, True)
        return MaestroOpenerInfo()


bundle_api = _MyAPI()
