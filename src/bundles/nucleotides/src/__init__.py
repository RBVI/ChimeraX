# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    # Override method for starting tool
    @staticmethod
    def start_tool(session, tool_name, **kw):
        from .tool import NucleotidesTool
        return NucleotidesTool(session, tool_name, **kw)

    # Override method for registering commands
    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        from chimerax.core.commands import register
        base_cmd = "nucleotides"
        for subcmd in ("", " style", " style list", " style delete", " ndbcolor"):
            cmd_name = base_cmd + subcmd
            func_name = cmd_name.replace(' ', '_')
            func = getattr(cmd, func_name)
            desc = getattr(cmd, func_name + "_desc")
            register(cmd_name, desc, func)
        from chimerax.core.commands import create_alias
        create_alias("~" + base_cmd, base_cmd + " atoms $*", logger=logger)


bundle_api = _MyAPI()
