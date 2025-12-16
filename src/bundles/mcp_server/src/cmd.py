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

    # Check Claude Desktop configuration file
    config_status = _check_claude_configuration(session)

    # Check REST server status and configuration
    rest_server = _get_rest_server()
    if rest_server is not None:
        port = rest_server.server_address[1]
        json_enabled = getattr(rest_server, "json", False)
        log_enabled = rest_server.log

        rest_status = f"REST server is running on port {port}<br>"
        if not (json_enabled and log_enabled):
            rest_status += "<br>&nbsp;&nbsp;💡 Run 'mcp start' to enable JSON and logging required MCP"
    else:
        rest_status = f"REST server is not running, start it with ChimeraX command 'mcp start'"

    status_info = f"{config_status}<br><br>{rest_status}<br><br>"
    session.logger.info(status_info, is_html=True)

    return status_info

def _check_claude_configuration(session):
    try:
        msg = mcp_setup(session, check_only = True)
    except UserError as e:
        msg = str(e)
    return msg

def mcp_setup(session, config_file_name = 'claude_desktop_config.json', check_only = False):
    "Write Claude Desktop configuration file to allow it to control ChimeraX using MCP."

    from sys import platform
    if platform == 'darwin':
        from os.path import expanduser
        config_dir = expanduser('~/Library/Application Support/Claude')
    elif platform == 'win32':
        from os import environ
        appdata_dir = environ.get('APPDATA')
        if not appdata_dir:
            from chimerax.core.errors import UserError
            raise UserError('Could not determine Claude Desktop configuration directory because APPDATA environment variable not set')
        from os.path import join
        config_dir = join(appdata_dir, 'Claude')
    else:
        from chimerax.core.errors import UserError
        raise UserError(f'Location of Claude Desktop configuration directory on "{platform}" is unknown')

    from os.path import isdir
    if not isdir(config_dir):
        from chimerax.core.errors import UserError
        raise UserError(f'The Claude Desktop configuration directory "{config_dir}" does not exist.  You need to install Claude Desktop on your computer and run it before configuring it for use with ChimeraX.')

    from os.path import join, isfile
    config_path = join(config_dir, config_file_name)
    if isfile(config_path):
        with open(config_path, 'r') as f:
            import json
            config_data = json.load(f)
        mcp_config = config_data.get('mcpServers')
        if mcp_config:
            if 'chimerax' in mcp_config:
                command_path = mcp_config['chimerax']['command']
                config_chimerax_path = _chimerax_directory_from_executable(command_path)
                import sys
                current_chimerax_path = _chimerax_directory_from_executable(sys.executable)
                if config_chimerax_path == current_chimerax_path:
                    msg = f'The Claude Desktop configuration file "{config_path}" is already set up to use ChimeraX.'
                    if check_only:
                        return msg
                    from chimerax.core.errors import UserError
                    raise UserError(msg)
                msg = f'Claude is configured to use a different ChimeraX version {config_chimerax_path}.'
                if check_only:
                    return msg
                msg += f' The configuration will be updated to use the ChimeraX you are now running {current_chimerax_path}'
                session.logger.info(msg)
        else:
            config_data['mcpServers'] = {}
    else:
        config_data = {'mcpServers': {}}

    if check_only:
        return 'Claude Desktop is not configured to use ChimeraX.  Run the ChimeraX command "mcp setup" to write the Claude Desktop configuration file.'

    config_data['mcpServers']['chimerax'] = _mcp_configuration_data()

    with open(config_path, 'w') as f:
        import json
        json.dump(config_data, f, indent=4)

    session.logger.info(f'Updated Claude Desktop configuration file {config_path} to use ChimeraX')

def _mcp_configuration_data():
    # Get the bridge script path
    from os.path import dirname, join, realpath
    bundle_dir = dirname(__file__)
    bridge_path = join(bundle_dir, "chimerax_mcp_bridge.py")

    # Get ChimeraX's Python executable path
    from sys import executable, platform, version_info
    python_executable_dir = dirname(realpath(executable))
    if platform == 'darwin':
        # On Mac the ChimeraX executable is in Contents/MacOS but the Python executable is in Contents/bin
        python_executable_dir = join(dirname(python_executable_dir), 'bin')
    python_executable_name = 'python.exe' if platform == 'win32' else f'python{version_info.major}.{version_info.minor}'
    python_path = join(python_executable_dir, python_executable_name)
    
    mcp_config = {
        'command': python_path,
        'args': [bridge_path]
    }
    return mcp_config

def _chimerax_directory_from_executable(exec_path):
    from sys import platform
    num_levels = {'darwin': 3, 'win32': 2}.get(platform, 2)
    path = exec_path
    from os.path import dirname
    for count in range(num_levels):
        path = dirname(path)
    return path
        
mcp_start_desc = CmdDesc(
    optional=[("port", IntArg)],
    synopsis="Start REST server for MCP bridge connections (default port: 8080)",
)

mcp_stop_desc = CmdDesc(synopsis="Stop the REST server")

mcp_info_desc = CmdDesc(synopsis="Show MCP bridge status and configuration")

mcp_setup_desc = CmdDesc(synopsis="Write Claude Desktop configuration file to allow it to control ChimeraX using MCP.")


def register_commands(logger):
    """Register MCP commands with ChimeraX"""
    register("mcp start", mcp_start_desc, mcp_start, logger=logger)
    register("mcp stop", mcp_stop_desc, mcp_stop, logger=logger)
    register("mcp info", mcp_info_desc, mcp_info, logger=logger)
    register("mcp setup", mcp_setup_desc, mcp_setup, logger=logger)
