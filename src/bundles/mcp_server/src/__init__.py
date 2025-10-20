__version__ = "0.1.0"

from chimerax.core.toolshed import BundleAPI


class _MCPServerBundleAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd

        cmd.register_commands(logger)


bundle_api = _MCPServerBundleAPI()

