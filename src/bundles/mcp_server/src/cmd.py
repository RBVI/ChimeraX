from chimerax.core.commands import CmdDesc, register, IntArg, BoolArg, EnumOf
from chimerax.core.errors import UserError


# Table of MCP-capable AI assistants ("hosts") this bundle knows how to
# configure. Each entry describes:
#   label:           human-readable name used in messages
#   config_dirs:     per-platform directory that contains the config file;
#                    the directory must exist (= the host is installed/has
#                    been launched at least once) before we will write to it.
#                    On Windows, values that start with "~" are expanded
#                    against %USERPROFILE% (via os.path.expanduser); other
#                    values are joined with %APPDATA%.
#   filename:        name of the config file in that directory
#   schema:          top-level JSON key under which MCP server entries live
#                    ("mcpServers" for Claude Desktop / Cursor, "servers"
#                    for VS Code Copilot)
#   needs_type_stdio: whether each server entry must include "type": "stdio"
#                    (true for VS Code Copilot's mcp.json schema)
_HOSTS = {
    "claude": {
        "label": "Claude Desktop",
        "config_dirs": {
            "darwin": "~/Library/Application Support/Claude",
            "win32":  "Claude",  # joined with %APPDATA%
        },
        "filename": "claude_desktop_config.json",
        "schema":   "mcpServers",
        "needs_type_stdio": False,
    },
    "cursor": {
        "label": "Cursor",
        "config_dirs": {
            # Cursor's docs state that ~/.cursor/mcp.json is the global
            # config location on all platforms, including Windows
            # (https://cursor.com/docs/mcp).
            "darwin": "~/.cursor",
            "linux":  "~/.cursor",
            "win32":  "~/.cursor",  # expanded against %USERPROFILE%
        },
        "filename": "mcp.json",
        "schema":   "mcpServers",
        "needs_type_stdio": False,
    },
    "vscode": {
        "label": "VS Code Copilot",
        "config_dirs": {
            "darwin": "~/Library/Application Support/Code/User",
            "linux":  "~/.config/Code/User",
            "win32":  "Code/User",  # joined with %APPDATA%
        },
        "filename": "mcp.json",
        "schema":   "servers",
        "needs_type_stdio": True,
    },
}

_HOST_CHOICES = list(_HOSTS.keys()) + ["all"]


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
    """Show MCP server status and AI-assistant host configuration"""

    # Per-host configuration status
    host_lines = []
    for host_id in _HOSTS:
        host_lines.append(_check_host_configuration(host_id))
    config_status = "<br>".join(host_lines)

    # Check REST server status and configuration
    rest_server = _get_rest_server()
    if rest_server is not None:
        port = rest_server.server_address[1]
        json_enabled = getattr(rest_server, "json", False)
        log_enabled = rest_server.log

        rest_status = f"REST server is running on port {port}<br>"
        if not (json_enabled and log_enabled):
            rest_status += "<br>&nbsp;&nbsp;💡 Run 'mcp start' to enable JSON and logging required by MCP"
    else:
        rest_status = f"REST server is not running, start it with ChimeraX command 'mcp start'"

    status_info = f"{config_status}<br><br>{rest_status}<br><br>"
    session.logger.info(status_info, is_html=True)

    return status_info


def _check_host_configuration(host_id):
    """Return a single-line status string for the given AI-assistant host."""
    host = _HOSTS[host_id]
    label = host["label"]
    try:
        msg = _setup_host(None, host_id, check_only=True)
        return f"{label}: {msg}"
    except UserError as e:
        return f"{label}: {str(e)}"


def _resolve_config_dir(host):
    """Resolve the platform-specific config directory for a host descriptor.

    Returns the absolute path, or raises UserError if this platform is not
    supported for that host.
    """
    from sys import platform
    config_dirs = host["config_dirs"]
    if platform == 'darwin':
        if 'darwin' not in config_dirs:
            raise UserError(
                f'{host["label"]} is not supported on macOS by mcp setup.')
        from os.path import expanduser
        return expanduser(config_dirs['darwin'])
    elif platform == 'win32':
        if 'win32' not in config_dirs:
            raise UserError(
                f'{host["label"]} is not supported on Windows by mcp setup.')
        win_dir = config_dirs['win32']
        # Some hosts (e.g. Cursor) put their global config under the user
        # profile (~/.cursor/mcp.json on all platforms), not under %APPDATA%.
        # Treat values starting with "~" as home-relative and let
        # os.path.expanduser pick up %USERPROFILE%; everything else is a
        # subdirectory under %APPDATA%.
        if win_dir.startswith('~'):
            from os.path import expanduser, normpath
            return normpath(expanduser(win_dir))
        from os import environ
        appdata_dir = environ.get('APPDATA')
        if not appdata_dir:
            raise UserError(
                f'Could not determine {host["label"]} configuration directory '
                f'because APPDATA environment variable not set')
        from os.path import join, normpath
        return normpath(join(appdata_dir, win_dir))
    else:
        # Linux / other
        linux_dir = config_dirs.get('linux')
        if linux_dir is None:
            raise UserError(
                f'Location of {host["label"]} configuration directory on '
                f'"{platform}" is unknown')
        from os.path import expanduser
        return expanduser(linux_dir)


def mcp_setup(session, host="claude", check_only=False):
    """Write a configuration file for an MCP-capable AI assistant so it can
    control ChimeraX using MCP.

    host: one of 'claude' (Claude Desktop, default), 'cursor', 'vscode'
          (VS Code Copilot), or 'all' to configure every host whose
          configuration directory is present on this machine.
    """
    if host == "all":
        results = []
        for host_id in _HOSTS:
            try:
                msg = _setup_host(session, host_id, check_only=check_only)
                if msg is not None:
                    results.append(f"{_HOSTS[host_id]['label']}: {msg}")
            except UserError as e:
                results.append(f"{_HOSTS[host_id]['label']}: {e}")
        if check_only:
            return "\n".join(results)
        for r in results:
            session.logger.info(r)
        return

    if host not in _HOSTS:
        raise UserError(
            f'Unknown host "{host}". Choose one of: '
            f'{", ".join(sorted(_HOSTS.keys()))}, all')

    return _setup_host(session, host, check_only=check_only)


def _load_host_config(path):
    """Read a host's MCP config file, tolerating empty files and JSONC.

    Cursor and VS Code both accept JSONC (// line comments, /* */ block
    comments, and trailing commas) for their mcp.json. ChimeraX does not
    bundle a JSONC parser, so we strip those constructs ourselves before
    a second parse attempt.

    Returns (config_data, had_jsonc).
    Raises ValueError with a friendly message if the file cannot be parsed
    even after JSONC cleanup.
    """
    import json
    import re
    with open(path, 'r') as f:
        text = f.read()
    if not text.strip():
        return {}, False
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        # Block comments first (so a // inside /* ... */ doesn't confuse us).
        cleaned = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        # Line comments. The (^|[^:]) guard avoids stripping `://` inside
        # URLs (e.g. Cursor entries with "url": "http://...").
        cleaned = re.sub(r"(^|[^:])//[^\n]*", r"\1", cleaned)
        # Trailing commas before } or ].
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned), True
        except json.JSONDecodeError as e:
            raise ValueError(
                f'Could not parse "{path}" as JSON or JSONC: {e}')


def _setup_host(session, host_id, check_only=False):
    """Write (or check) the configuration file for one host.

    When check_only is True, returns a status string and never modifies any
    files (session may be None in this mode).
    """
    host = _HOSTS[host_id]
    config_dir = _resolve_config_dir(host)

    from os.path import isdir, isfile, join
    if not isdir(config_dir):
        raise UserError(
            f'The {host["label"]} configuration directory "{config_dir}" '
            f'does not exist. You need to install {host["label"]} on your '
            f'computer and run it once before configuring it for use with '
            f'ChimeraX.')

    config_path = join(config_dir, host["filename"])
    schema_key = host["schema"]

    had_jsonc = False
    config_existed = isfile(config_path)
    if config_existed:
        try:
            config_data, had_jsonc = _load_host_config(config_path)
        except ValueError as e:
            if check_only:
                return f'existing config file could not be parsed: {e}'
            raise UserError(
                f'{host["label"]} configuration file "{config_path}" could '
                f'not be parsed: {e}. Fix or remove it before running '
                f'"mcp setup".')
        if not config_data:
            config_data = {schema_key: {}}
        servers = config_data.get(schema_key)
        if servers:
            if 'chimerax' in servers:
                command_path = servers['chimerax'].get('command', '')
                config_chimerax_path = _chimerax_directory_from_executable(command_path)
                import sys
                current_chimerax_path = _chimerax_directory_from_executable(sys.executable)
                if config_chimerax_path == current_chimerax_path:
                    msg = f'configured to use this ChimeraX (file "{config_path}")'
                    if check_only:
                        return msg
                    raise UserError(
                        f'The {host["label"]} configuration file '
                        f'"{config_path}" is already set up to use ChimeraX.')
                msg = (f'configured to use a different ChimeraX version '
                       f'{config_chimerax_path}')
                if check_only:
                    return msg
                session.logger.info(
                    f'{host["label"]} {msg}. The configuration will be '
                    f'updated to use the ChimeraX you are now running '
                    f'{current_chimerax_path}')
        else:
            config_data[schema_key] = {}
    else:
        config_data = {schema_key: {}}

    if check_only:
        return (f'not configured to use ChimeraX. Run the ChimeraX command '
                f'"mcp setup host {host_id}" to write the {host["label"]} '
                f'configuration file.')

    config_data[schema_key]['chimerax'] = _mcp_configuration_data(host)

    # Snapshot the previous file before we overwrite it, so the user can
    # restore comments / original formatting if anything looks off.
    backup_path = None
    if config_existed:
        import shutil
        backup_path = config_path + '.bak'
        try:
            shutil.copyfile(config_path, backup_path)
        except OSError as e:
            session.logger.warning(
                f'Could not create backup "{backup_path}" before rewriting '
                f'{host["label"]} config: {e}')
            backup_path = None

    with open(config_path, 'w') as f:
        import json
        json.dump(config_data, f, indent=4)

    session.logger.info(
        f'Updated {host["label"]} configuration file {config_path} to use '
        f'ChimeraX')

    if had_jsonc:
        bak_msg = (f' The previous contents are saved at "{backup_path}".'
                   if backup_path else '')
        session.logger.warning(
            f'{host["label"]} configuration file "{config_path}" contained '
            f'comments or trailing commas; ChimeraX has rewritten it as '
            f'strict JSON and comments are no longer preserved.{bak_msg}')


def _mcp_configuration_data(host):
    """Build the per-host MCP server entry for ChimeraX.

    Adds "type": "stdio" for hosts (e.g. VS Code Copilot) whose schema
    requires it.
    """
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
    if host.get("needs_type_stdio"):
        # Insert "type" first for nicer JSON ordering
        mcp_config = {'type': 'stdio', **mcp_config}
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

mcp_info_desc = CmdDesc(synopsis="Show MCP bridge status and AI-assistant host configuration")

mcp_setup_desc = CmdDesc(
    keyword=[("host", EnumOf(_HOST_CHOICES))],
    synopsis="Write a configuration file for an MCP-capable AI assistant "
             "(Claude Desktop, Cursor, VS Code Copilot) so it can control "
             "ChimeraX using MCP.",
)


def register_commands(logger):
    """Register MCP commands with ChimeraX"""
    register("mcp start", mcp_start_desc, mcp_start, logger=logger)
    register("mcp stop", mcp_stop_desc, mcp_stop, logger=logger)
    register("mcp info", mcp_info_desc, mcp_info, logger=logger)
    register("mcp setup", mcp_setup_desc, mcp_setup, logger=logger)
