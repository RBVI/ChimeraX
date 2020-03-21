# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def open_file(session, path, file_name, format_name,
                  auto_style=True, atomic=True):
        if format_name == "mae":
            from .io import open_mae
            return open_mae(session, path, file_name, auto_style, atomic)
        else:
            raise ValueError("unsupported format: %s" % format_name)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_cmd import OpenerInfo
        class MaestroOpenerInfo(OpenerInfo):
            def open(self, session, path, file_name, **kw):
                from .io import open_mae
                return open_mae(session, path, file_name, True, True)
        return MaestroOpenerInfo()


bundle_api = _MyAPI()
