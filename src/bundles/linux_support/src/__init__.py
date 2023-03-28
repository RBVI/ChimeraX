# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        from chimerax.core.commands import register
        base_cmd = "linux"
        # for subcmd in ("", " style", " style list", " style delete"):
        for subcmd in (" xdg-install", " xdg-uninstall", " flatpak-files"):
            cmd_name = base_cmd + subcmd
            func_name = cmd_name.replace(' ', '_').replace('-', '_')
            func = getattr(cmd, func_name)
            desc = getattr(cmd, func_name + "_desc")
            register(cmd_name, desc, func)


bundle_api = _MyAPI()
