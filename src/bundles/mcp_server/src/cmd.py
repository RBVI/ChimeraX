from chimerax.core.commands import CmdDesc, register, IntArg, BoolArg
from chimerax.core.errors import UserError


def mcp_start(session, port=3001):
    """Start the MCP server"""
    if not hasattr(session, 'mcp_server'):
        raise UserError("MCP server not initialized")

    success, message = session.mcp_server.start(port)
    session.logger.info(message)
    return success, message


def mcp_stop(session):
    """Stop the MCP server"""
    if not hasattr(session, 'mcp_server'):
        raise UserError("MCP server not initialized")

    success, message = session.mcp_server.stop()
    session.logger.info(message)
    return success, message


def mcp_status(session):
    """Show MCP server status"""
    if not hasattr(session, 'mcp_server'):
        raise UserError("MCP server not initialized")

    success, message = session.mcp_server.status()
    settings = session.mcp_server.settings

    status_info = f"{message}\n"
    status_info += f"Settings:\n"
    status_info += f"  Auto-start: {settings.auto_start}\n"
    status_info += f"  Default port: {settings.port}\n"
    status_info += f"  Log to ChimeraX: {settings.log_to_chimerax}"

    session.logger.info(status_info)
    return success, status_info


mcp_start_desc = CmdDesc(
    optional=[("port", IntArg)],
    synopsis="Start MCP server on specified port (default: 3001)"
)

mcp_stop_desc = CmdDesc(
    synopsis="Stop the MCP server"
)

mcp_status_desc = CmdDesc(
    synopsis="Show MCP server status"
)


def register_commands(logger):
    """Register MCP commands with ChimeraX"""
    register("mcp start", mcp_start_desc, mcp_start, logger=logger)
    register("mcp stop", mcp_stop_desc, mcp_stop, logger=logger)
    register("mcp status", mcp_status_desc, mcp_status, logger=logger)