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

        # Register settings in UI if GUI is available
        if session.ui.is_gui:
            from . import settings
            session.ui.triggers.add_handler('ready',
                lambda *args, ses=session: settings.register_settings_options(ses))

        # Auto-start server if enabled in settings
        if session.mcp_server.settings.auto_start:
            port = session.mcp_server.settings.port
            success, message = session.mcp_server.start(port)
            if success:
                session.logger.info(f"Auto-started MCP server: {message}")
            else:
                session.logger.warning(f"Failed to auto-start MCP server: {message}")

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_commands(logger)


bundle_api = _MCPServerBundleAPI()