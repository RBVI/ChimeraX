# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_command(ci, logger)


bundle_api = _MyAPI()
