# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI
from .job import BlastPDBJob


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
    def register_command(command_name):
        from . import cmd
        from chimerax.core.commands import register
        if command_name == "blastpdb":
            register(command_name, cmd.blastpdb_desc, cmd.blastpdb)
        elif command_name == "ccd":
            register(command_name, cmd.ccd_desc, cmd.ccd)

bundle_api = _MyAPI()
