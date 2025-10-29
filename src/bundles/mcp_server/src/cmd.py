from chimerax.core.commands import CmdDesc, register, IntArg, BoolArg
from chimerax.core.errors import UserError


def mcp_start(session, port=8080):
    """Start or reconfigure REST server for MCP bridge connections"""
    from chimerax.core.commands import run

    try:
        # Check if REST server is already running
        try:
            run(session, "remotecontrol rest port")
            # Server is running, check if we need to reconfigure it
            rest_server = _get_rest_server()
            if rest_server is not None:
                current_port = rest_server.server_address[1]

                # Check if we need to reconfigure for MCP compatibility
                needs_reconfigure = False

                # Check JSON setting
                if not getattr(rest_server, "json", False):
                    rest_server.json = True
                    needs_reconfigure = True

                # Check log setting
                if not rest_server.log:
                    rest_server.log = True
                    needs_reconfigure = True

                if current_port == port:
                    if needs_reconfigure:
                        message = f"REST server on port {port} reconfigured for MCP bridge compatibility (JSON: enabled, Log: enabled)"
                    else:
                        message = f"REST server on port {port} already configured for MCP bridge connections"
                    session.logger.info(message)
                    return True, message
                else:
                    # Different port requested, stop and restart
                    run(session, "remotecontrol rest stop quiet true")
                    run(
                        session,
                        f"remotecontrol rest start port {port} json true log true",
                    )
                    message = f"REST server restarted on port {port} for MCP bridge connections"
                    session.logger.info(message)
                    return True, message
            else:
                # Server reference lost, try to restart
                run(session, f"remotecontrol rest start port {port} json true log true")
                message = (
                    f"REST server started on port {port} for MCP bridge connections"
                )
                session.logger.info(message)
                return True, message

        except:
            # Server not running, start it
            run(session, f"remotecontrol rest start port {port} json true log true")
            message = f"REST server started on port {port} for MCP bridge connections"
            session.logger.info(message)
            return True, message

    except Exception as e:
        error_msg = f"Failed to start/configure REST server: {e}"
        session.logger.error(error_msg)
        return False, error_msg


def _get_rest_server():
    """Get the current REST server instance"""
    try:
        import chimerax.rest_server.cmd as rest_cmd

        return rest_cmd._get_server()
    except:
        return None


def mcp_stop(session):
    """Stop the REST server"""
    from chimerax.core.commands import run

    try:
        result = run(session, "remotecontrol rest stop")
        message = "REST server stopped"
        session.logger.info(message)
        return True, message
    except Exception as e:
        error_msg = f"Failed to stop REST server: {e}"
        session.logger.error(error_msg)
        return False, error_msg


def mcp_info(session):
    """Show MCP server status and configuration"""
    from chimerax.core.commands import run
    import json
    import os
    import sys

    # Check REST server status and configuration
    rest_server = _get_rest_server()
    if rest_server is not None:
        port = rest_server.server_address[1]
        json_enabled = getattr(rest_server, "json", False)
        log_enabled = rest_server.log

        json_status = "✓ enabled" if json_enabled else "✗ disabled"
        log_status = "✓ enabled" if log_enabled else "✗ disabled"

        mcp_ready = json_enabled and log_enabled
        mcp_status = "✓ ready" if mcp_ready else "⚠ needs configuration"

        rest_status = f"REST server is running on port {port}<br>"
        rest_status += f"&nbsp;&nbsp;JSON output: {json_status}<br>"
        rest_status += f"&nbsp;&nbsp;Log capture: {log_status}<br>"
        rest_status += f"&nbsp;&nbsp;MCP compatibility: {mcp_status}"

        if not mcp_ready:
            rest_status += (
                "<br>&nbsp;&nbsp;💡 Run 'mcp start' to enable MCP compatibility"
            )
    else:
        rest_status = "REST server is not running"

    # Get the bridge script path
    bundle_dir = os.path.dirname(__file__)
    bridge_path = os.path.join(bundle_dir, "chimerax_mcp_bridge.py")

    # Get ChimeraX's Python executable path
    chimerax_python_dir = os.path.dirname(sys.executable)
    chimerax_python = (
        chimerax_python_dir + os.sep + "python3.11"
    )  # Ensure correct path format

    # Generate Claude configuration (no env needed - bridge auto-discovers)
    config_lines = [
        "{",
        '  "mcpServers": {',
        '    "chimerax": {',
        f'      "command": "{chimerax_python}",',
        f'      "args": ["{bridge_path}"]',
        "    }",
        "  }",
        "}",
    ]
    config_json = "\n".join(config_lines)

    status_info = f"{rest_status}<br><br>"
    status_info += f"MCP Bridge Setup:<br>"
    status_info += f"&nbsp;&nbsp;Bridge script: {bridge_path}<br>"
    status_info += f"&nbsp;&nbsp;ChimeraX Python: {chimerax_python}<br><br>"
    status_info += f"Claude Desktop Configuration:<br>"
    status_info += f"Copy this JSON to your Claude Desktop config file:<br><br>"
    status_info += f"<pre><code>{config_json}</code></pre><br>"
    if (
        rest_server is not None
        and getattr(rest_server, "json", False)
        and rest_server.log
    ):
        status_info += f"Usage: MCP bridge ready - start Claude Desktop to connect"
    else:
        status_info += f"Usage: Run 'mcp start' to configure REST server for MCP bridge connections"

    session.logger.info(status_info, is_html=True)
    return True, status_info


mcp_start_desc = CmdDesc(
    optional=[("port", IntArg)],
    synopsis="Start REST server for MCP bridge connections (default port: 8080)",
)

mcp_stop_desc = CmdDesc(synopsis="Stop the REST server")

mcp_info_desc = CmdDesc(synopsis="Show MCP bridge status and configuration")


def register_commands(logger):
    """Register MCP commands with ChimeraX"""
    register("mcp start", mcp_start_desc, mcp_start, logger=logger)
    register("mcp stop", mcp_stop_desc, mcp_stop, logger=logger)
    register("mcp info", mcp_info_desc, mcp_info, logger=logger)
