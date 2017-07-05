# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    # Override method for starting tool
    @staticmethod
    def start_tool(session, tool_name, **kw):
        from .tool import SampleTool
        return SampleTool(session, tool_name, **kw)

    # Override method for registering commands
    @staticmethod
    def register_command(command_name, logger):
        # We expect that there is a function in "cmd"
        # corresponding to every registered command
        # in "setup.py.in" and that they are named
        # identically (except with '_' replacing spaces)
        from . import cmd
        from chimerax.core.commands import register
        base_name = command_name.replace(" ", "_")
        func = getattr(cmd, base_name)
        desc = getattr(cmd, base_name + "_desc")
        register(command_name, desc, func)

    # Override method for opening file
    @staticmethod
    def open_file(session, stream, name):
        from .io import open_xyz
        return open_xyz(session, stream, name)


bundle_api = _MyAPI()
