# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        from . import _data
        if class_name == 'NucleotideState':
            return _data.NucleotideState
        if class_name == 'Params':
            return _data.Params
        return None

    @staticmethod
    def start_tool(session, tool_name, **kw):
        from .tool import NucleotidesTool
        return NucleotidesTool(session, tool_name, **kw)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        from chimerax.core.commands import register
        base_cmd = "nucleotides"
        # for subcmd in ("", " style", " style list", " style delete"):
        for subcmd in ("",):
            cmd_name = base_cmd + subcmd
            func_name = cmd_name.replace(' ', '_')
            func = getattr(cmd, func_name)
            desc = getattr(cmd, func_name + "_desc")
            register(cmd_name, desc, func)
        from chimerax.core.commands import create_alias
        create_alias("~" + base_cmd, base_cmd + " $* atoms", logger=logger)

    @staticmethod
    def run_provider(session, name, mgr, *, display_name=None):
        """Run toolbar provider"""
        from . import cmd
        cmd.run_provider(session, name, display_name)


bundle_api = _MyAPI()
