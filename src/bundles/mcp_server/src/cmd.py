from chimerax.core.commands import CmdDesc, register, IntArg
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
    session.logger.info(message)
    return success, message


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