__version__ = "0.1.0"

from chimerax.core.toolshed import BundleAPI


class _MCPServerBundleAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def get_class(class_name):
        if class_name == "MCPServer":
            from . import server
            return server.MCPServer

    @staticmethod
    def initialize(session, bundle_info):
        """Initialize MCP server manager in the session"""
        from .server import MCPServer
        session.mcp_server = MCPServer(session)

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_commands(logger)


bundle_api = _MCPServerBundleAPI()