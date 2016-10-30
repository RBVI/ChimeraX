# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):
    # FIXME: only implement methods that the metadata says should be there

    @staticmethod
    def start_tool(session, tool_name):
        # 'start_tool' is called to start an instance of the tool
        # If providing more than one tool in package,
        # look at 'tool_name' to see which is being started.
        raise NotImplementedError  # FIXME: remove method if unneeded
        from .tool import ToolUI
        # UI should register itself with tool state manager
        return ToolUI(session, tool_name)

    @staticmethod
    def register_command(command_name):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register(command_name,
                 cmd.stringdb_desc, cmd.stringdb)
        # TODO: Register more subcommands here

bundle_api = _MyAPI()
