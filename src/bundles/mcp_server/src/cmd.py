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

    import json
    import os

    success, message = session.mcp_server.status()
    settings = session.mcp_server.settings

    # Get the bridge script path
    bundle_dir = os.path.dirname(__file__)
    bridge_path = os.path.join(bundle_dir, "chimerax_mcp_bridge.js")

    # Generate Claude configuration
    config = {
        "mcpServers": {
            "chimerax": {
                "command": "node",
                "args": [bridge_path],
                "env": {
                    "CHIMERAX_MCP_HOST": "localhost",
                    "CHIMERAX_MCP_PORT": str(settings.port)
                }
            }
        }
    }

    # Format JSON with proper indentation that won't get stripped
    config_lines = [
        "{",
        '  "mcpServers": {',
        '    "chimerax": {',
        '      "command": "node",',
        f'      "args": ["{bridge_path}"],',
        '      "env": {',
        '        "CHIMERAX_MCP_HOST": "localhost",',
        f'        "CHIMERAX_MCP_PORT": "{settings.port}"',
        '      }',
        '    }',
        '  }',
        "}"
    ]
    config_json = "\n".join(config_lines)

    status_info = f"{message}<br><br>"
    status_info += f"Settings:<br>"
    status_info += f"&nbsp;&nbsp;Auto-start: {settings.auto_start}<br>"
    status_info += f"&nbsp;&nbsp;Default port: {settings.port}<br><br>"
    status_info += f"Claude Desktop Configuration:<br>"
    status_info += f"Copy this JSON to your Claude Desktop config file:<br><br>"
    status_info += f"<pre><code>{config_json}</code></pre><br>"
    status_info += f"Bridge script: {bridge_path}"

    session.logger.info(status_info, is_html=True)
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

mcp_setup_desc = CmdDesc(
    synopsis="Install MCP SDK and generate Claude configuration"
)


def mcp_setup(session):
    """Install MCP SDK and generate Claude configuration"""
    if not hasattr(session, 'mcp_server'):
        raise UserError("MCP server not initialized")

    import subprocess
    import json
    import os
    import sys

    try:
        # Get the bundle directory
        bundle_dir = os.path.dirname(__file__)
        bridge_path = os.path.join(bundle_dir, "chimerax_mcp_bridge.js")

        settings = session.mcp_server.settings
        port = settings.port

        # Ensure package.json exists
        package_json_path = os.path.join(bundle_dir, "package.json")
        if not os.path.exists(package_json_path):
            package_json = {
                "name": "chimerax-mcp-bridge",
                "version": "1.0.0",
                "type": "module",
                "description": "MCP bridge for ChimeraX",
                "main": "chimerax_mcp_bridge.js",
                "dependencies": {
                    "@modelcontextprotocol/sdk": "^0.5.0"
                },
                "engines": {
                    "node": ">=18.0.0"
                }
            }
            with open(package_json_path, 'w') as f:
                json.dump(package_json, f, indent=2)
            session.logger.info("Created package.json")

        # Install MCP SDK
        session.logger.info("Installing @modelcontextprotocol/sdk...")
        try:
            # Try to install in the bundle directory
            result = subprocess.run([
                "npm", "install", "@modelcontextprotocol/sdk"
            ], cwd=bundle_dir, capture_output=True, text=True, check=True)
            session.logger.info("MCP SDK installed successfully")
        except subprocess.CalledProcessError as e:
            session.logger.error(f"Failed to install MCP SDK: {e.stderr}")
            return False, f"Failed to install MCP SDK: {e.stderr}"
        except FileNotFoundError:
            session.logger.error("npm not found. Please install Node.js and npm first.")
            return False, "npm not found. Please install Node.js and npm first."

        # Generate Claude configuration
        config = {
            "mcpServers": {
                "chimerax": {
                    "command": "node",
                    "args": [bridge_path],
                    "env": {
                        "CHIMERAX_MCP_HOST": "localhost",
                        "CHIMERAX_MCP_PORT": str(port)
                    }
                }
            }
        }

        config_json = json.dumps(config, indent=2)

        message = f"""MCP setup completed successfully!

1. MCP SDK installed in: {bundle_dir}

2. Add this configuration to your Claude Desktop config file:

{config_json}

3. To use:
   - Start ChimeraX
   - Run: mcp start {port}
   - Start Claude Desktop (it will automatically connect)

4. Bridge script location: {bridge_path}

Note: Make sure Node.js is installed and accessible in your PATH."""

        session.logger.info(message)
        return True, message

    except Exception as e:
        error_msg = f"Setup failed: {e}"
        session.logger.error(error_msg)
        return False, error_msg


def register_commands(logger):
    """Register MCP commands with ChimeraX"""
    register("mcp start", mcp_start_desc, mcp_start, logger=logger)
    register("mcp stop", mcp_stop_desc, mcp_stop, logger=logger)
    register("mcp status", mcp_status_desc, mcp_status, logger=logger)
    register("mcp setup", mcp_setup_desc, mcp_setup, logger=logger)