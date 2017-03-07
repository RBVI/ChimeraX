# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI
from .job import BlastProteinJob


class _MyAPI(BundleAPI):
    @staticmethod
    def get_class(class_name):
        if class_name == 'ToolUI':
            from . import tool
            return tool.ToolUI
        return None

    @staticmethod
    def start_tool(session, tool_name, **kw):
        from .tool import ToolUI
        return ToolUI(session, tool_name, **kw)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        if command_name == "blastprotein":
            from chimerax.core.commands import register
            register(command_name, cmd.blastprotein_desc, cmd.blastprotein, logger=logger)
        elif command_name == "blastpdb":
            from chimerax.core.commands import create_alias
            create_alias(command_name, "blastprotein $*", logger=logger)

bundle_api = _MyAPI()
