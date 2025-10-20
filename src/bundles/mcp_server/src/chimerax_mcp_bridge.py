#!/usr/bin/env python3

"""
ChimeraX MCP Bridge (Python)

This bridges Claude (via MCP protocol) to ChimeraX REST servers.
Uses the official MCP Python SDK for robust protocol handling.
Supports M-to-N architecture: multiple agents controlling multiple ChimeraX instances.

Features:
- Auto-start ChimeraX instances when needed
- Multi-instance support with port-based session IDs
- Cross-platform daemon mode (double-fork on Unix, DETACHED_PROCESS on Windows)
- Instance discovery and management
- Session-aware tool routing

Prerequisites:
1. Install the ChimeraX MCP server bundle in ChimeraX
2. ChimeraX will be auto-started when needed

Usage with Claude Desktop:
Add to your claude_desktop_config.json:
{
  "mcpServers": {
    "chimerax": {
      "command": "python",
      "args": ["/path/to/chimerax_mcp_bridge.py"],
      "env": {
        "CHIMERAX_REST_HOST": "localhost",
        "CHIMERAX_REST_PORT": "8080"
      }
    }
  }
}

Multi-Instance Usage:
- Tools accept optional session_id parameter (port number)
- Use start_new_chimerax_session() to create new instances
- Use list_chimerax_instances() to see all running instances
- Use set_default_session() to change default target
"""

import os
import sys
import asyncio
import aiohttp
import tempfile
import base64
import subprocess
import time
import platform
from typing import Optional

# ChimeraX connection settings
CHIMERAX_HOST = 'localhost'
DEFAULT_CHIMERAX_PORT = 8080

# Global state for managing multiple ChimeraX instances
_instances = {}  # port -> instance info
_default_port = DEFAULT_CHIMERAX_PORT

from mcp.server.fastmcp import FastMCP

def find_chimerax_executable():
    """Find ChimeraX executable, starting from bridge script location"""

    # First, try to find ChimeraX relative to this bridge script
    bridge_path = os.path.abspath(__file__)

    # Work backwards from bridge to find ChimeraX executable
    current_dir = os.path.dirname(bridge_path)

    # Look for ChimeraX structure patterns
    for _ in range(10):  # Limit search depth
        current_dir = os.path.dirname(current_dir)

        # Check for macOS app bundle structure
        if current_dir.endswith('.app'):
            chimerax_exe = os.path.join(current_dir, 'Contents', 'bin', 'ChimeraX')
            if os.path.exists(chimerax_exe):
                return chimerax_exe

        # For development builds, look for ChimeraX.app at this level
        chimerax_app = os.path.join(current_dir, 'ChimeraX.app')
        if os.path.exists(chimerax_app):
            chimerax_exe = os.path.join(chimerax_app, 'Contents', 'bin', 'ChimeraX')
            if os.path.exists(chimerax_exe):
                return chimerax_exe

        # Check for Linux/Windows structure - bin directory at this level
        bin_dir = os.path.join(current_dir, 'bin')
        if os.path.exists(bin_dir):
            # Try different executable names
            for exe_name in ['ChimeraX', 'chimerax', 'ChimeraX.exe']:
                chimerax_exe = os.path.join(bin_dir, exe_name)
                if os.path.exists(chimerax_exe):
                    return chimerax_exe

    # Fallback to system paths if relative detection fails
    system = platform.system()

    if system == "Darwin":  # macOS
        paths = [
            "/Applications/ChimeraX.app/Contents/bin/ChimeraX",
            "/Applications/ChimeraX-Daily.app/Contents/bin/ChimeraX",
            "~/Applications/ChimeraX.app/Contents/bin/ChimeraX"
        ]
    elif system == "Windows":
        paths = [
            "C:\\Program Files\\ChimeraX\\bin\\ChimeraX.exe",
            "C:\\Program Files (x86)\\ChimeraX\\bin\\ChimeraX.exe",
            os.path.expanduser("~\\AppData\\Local\\ChimeraX\\bin\\ChimeraX.exe")
        ]
    else:  # Linux
        paths = [
            "/usr/local/bin/chimerax",
            "/opt/chimerax/bin/chimerax",
            os.path.expanduser("~/chimerax/bin/chimerax")
        ]

    # Also check PATH
    try:
        result = subprocess.run(["which", "chimerax"], capture_output=True, text=True)
        if result.returncode == 0:
            paths.insert(0, result.stdout.strip())
    except:
        pass

    for path in paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            return expanded_path

    return None

def find_available_port(start_port: int = 8080) -> int:
    """Find an available port starting from start_port"""
    import socket
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((CHIMERAX_HOST, port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

async def is_chimerax_running(port: int = None) -> bool:
    """Check if ChimeraX REST server is running on specified port"""
    if port is None:
        port = _default_port

    try:
        session = await get_session()
        # Use a simple GET request to a known static file
        url = f"http://{CHIMERAX_HOST}:{port}/cmdline.html"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=1)) as response:
            return response.status == 200
    except:
        return False

def get_chimerax_url(port: int = None) -> str:
    """Get ChimeraX REST API base URL for specified port"""
    if port is None:
        port = _default_port
    return f"http://{CHIMERAX_HOST}:{port}"

async def list_running_instances() -> dict:
    """List all running ChimeraX instances"""
    running = {}

    # Check known instances
    for port, info in _instances.items():
        if await is_chimerax_running(port):
            running[port] = info

    # Also scan common ports for unknown instances
    for port in range(8080, 8090):
        if port not in _instances and await is_chimerax_running(port):
            running[port] = {"status": "discovered", "started_by": "unknown"}

    return running

def start_chimerax_daemon(port: int):
    """Start ChimeraX in daemon mode (double-fork) on specified port"""
    chimerax_path = find_chimerax_executable()
    if not chimerax_path:
        return False

    try:
        # Double-fork to create daemon process
        pid = os.fork()
        if pid > 0:
            # Parent process - wait for first child and return
            os.waitpid(pid, 0)
            return True

        # First child
        os.setsid()  # Create new session
        pid = os.fork()
        if pid > 0:
            # First child exits
            sys.exit(0)

        # Second child (daemon)
        os.chdir("/")
        os.umask(0)

        # Redirect stdin/stdout/stderr to /dev/null
        with open(os.devnull, 'r') as devnull:
            os.dup2(devnull.fileno(), sys.stdin.fileno())
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())

        # Start ChimeraX with REST server on specified port
        cmd = [chimerax_path, "--cmd", f"remotecontrol rest start port {port} json true log true"]
        os.execv(chimerax_path, cmd)

    except OSError:
        return False

async def check_existing_rest_server() -> tuple[bool, int]:
    """Check if ChimeraX already has a REST server running, returns (has_server, port)"""

    # Try to find any running ChimeraX instances first
    for check_port in [_default_port] + [8081, 8082, 8083, 7955, 9000]:
        if await is_chimerax_running(check_port):
            try:
                # Found ChimeraX, check if it has REST server info
                session = await get_session()
                params = {'command': 'remotecontrol rest port'}
                url = f"http://{CHIMERAX_HOST}:{check_port}/run"

                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        result = await response.text()
                        # Parse the response to extract port info
                        if "listening on" in result:
                            # Extract port from message like "REST server is listening on host localhost port 8080"
                            import re
                            port_match = re.search(r'port (\d+)', result)
                            if port_match:
                                rest_port = int(port_match.group(1))
                                print(f"Found existing REST server on port {rest_port}", file=sys.stderr)
                                return True, rest_port
                        elif "not running" in result:
                            print(f"ChimeraX on port {check_port} has no REST server running", file=sys.stderr)
                            return False, check_port
            except:
                continue

    return False, _default_port

async def start_chimerax(port: int = None, session_name: str = None, force_new: bool = False) -> tuple[bool, int]:
    """Start ChimeraX if not running, returns (success, port)"""
    if port is None:
        port = _default_port

    # First check if there's already a ChimeraX with REST server running (unless forcing new)
    if not force_new:
        has_rest, existing_port = await check_existing_rest_server()
        if has_rest:
            print(f"Using existing ChimeraX REST server on port {existing_port}", file=sys.stderr)
            return True, existing_port

    # Check if already running on this port
    if await is_chimerax_running(port):
        return True, port

    # If port is taken but not by ChimeraX, find a new one
    if port != _default_port:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((CHIMERAX_HOST, port))
        except OSError:
            port = find_available_port(port)

    chimerax_path = find_chimerax_executable()
    if not chimerax_path:
        print("ChimeraX executable not found. Please install ChimeraX or add it to PATH.", file=sys.stderr)
        return False, port

    try:
        print(f"Starting ChimeraX from {chimerax_path} on port {port}...", file=sys.stderr)

        if platform.system() == "Windows":
            # Windows doesn't have fork, use subprocess with DETACHED_PROCESS
            cmd = [chimerax_path, "--cmd", f"remotecontrol rest start port {port} json true log true"]
            subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Unix-like systems: use double-fork
            if not start_chimerax_daemon(port):
                print("Failed to start ChimeraX daemon", file=sys.stderr)
                return False, port

        # Register the instance
        _instances[port] = {
            "status": "starting",
            "started_by": "bridge",
            "session_name": session_name or f"session_{port}",
            "start_time": time.time()
        }

        # Wait for ChimeraX to start (up to 30 seconds)
        for _ in range(30):
            await asyncio.sleep(1)
            if await is_chimerax_running(port):
                print(f"ChimeraX started successfully on port {port}!", file=sys.stderr)
                _instances[port]["status"] = "running"
                return True, port

        print("Timeout waiting for ChimeraX to start", file=sys.stderr)
        _instances[port]["status"] = "failed"
        return False, port

    except Exception as e:
        print(f"Failed to start ChimeraX: {e}", file=sys.stderr)
        if port in _instances:
            _instances[port]["status"] = "failed"
        return False, port


# Create FastMCP server instance
mcp = FastMCP("ChimeraX Bridge")

def get_docs_path():
    """Get the path to ChimeraX documentation"""
    # Find docs relative to bridge script location
    bridge_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(bridge_path)

    # Look for docs in ChimeraX structure
    for _ in range(10):
        current_dir = os.path.dirname(current_dir)

        # Check for installed docs location
        docs_path = os.path.join(current_dir, 'Contents', 'share', 'docs')
        if os.path.exists(docs_path):
            return docs_path

        # Check for ChimeraX.app at this level
        chimerax_app = os.path.join(current_dir, 'ChimeraX.app')
        if os.path.exists(chimerax_app):
            docs_path = os.path.join(chimerax_app, 'Contents', 'share', 'docs')
            if os.path.exists(docs_path):
                return docs_path

    return None

def list_available_commands():
    """List all available ChimeraX commands with documentation"""
    docs_path = get_docs_path()
    if not docs_path:
        return []

    commands_path = os.path.join(docs_path, 'user', 'commands')
    if not os.path.exists(commands_path):
        return []

    commands = []
    for filename in os.listdir(commands_path):
        if filename.endswith('.html'):
            command_name = filename[:-5]  # Remove .html
            commands.append(command_name)

    return sorted(commands)

def get_command_doc(command_name: str) -> str:
    """Get documentation for a specific command"""
    docs_path = get_docs_path()
    if not docs_path:
        return f"Documentation not found for command: {command_name}"

    doc_file = os.path.join(docs_path, 'user', 'commands', f'{command_name}.html')
    if not os.path.exists(doc_file):
        return f"No documentation found for command: {command_name}"

    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract just the useful content (remove HTML boilerplate)
        from html.parser import HTMLParser
        import re

        class DocExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.content = []
                self.in_body = False
                self.current_text = ""

            def handle_starttag(self, tag, attrs):
                if tag == 'body':
                    self.in_body = True
                elif tag in ['h3', 'p', 'li'] and self.in_body:
                    self.current_text += f"\n{tag.upper()}: "

            def handle_data(self, data):
                if self.in_body:
                    clean_data = re.sub(r'\s+', ' ', data.strip())
                    if clean_data:
                        self.current_text += clean_data + " "

            def handle_endtag(self, tag):
                if tag == 'body':
                    self.in_body = False

        parser = DocExtractor()
        parser.feed(content)

        # Clean up the extracted text
        doc_text = parser.current_text
        doc_text = re.sub(r'\n+', '\n', doc_text)
        doc_text = re.sub(r'\s+', ' ', doc_text)

        return f"ChimeraX Command: {command_name}\n\n{doc_text[:2000]}..."  # Limit length

    except Exception as e:
        return f"Error reading documentation for {command_name}: {e}"

# Global aiohttp session for REST API calls
_session: Optional[aiohttp.ClientSession] = None

async def get_session() -> aiohttp.ClientSession:
    """Get or create aiohttp session"""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session

async def find_best_chimerax_instance() -> int:
    """Find the best ChimeraX instance to use (prefer existing, fallback to default port)"""
    global _default_port

    # First check if default port is running
    if await is_chimerax_running(_default_port):
        print(f"Using default ChimeraX instance on port {_default_port}", file=sys.stderr)
        return _default_port

    # Quick scan for any running ChimeraX instances (common ports only)
    common_ports = [8081, 8082, 8083, 7955, 9000]  # Common alternatives
    for port in common_ports:
        if port == _default_port:
            continue  # Already checked

        if await is_chimerax_running(port):
            # Found one! Update our default for future use
            print(f"Found ChimeraX instance on port {port}, using as default", file=sys.stderr)
            _default_port = port
            return port

    # No instances found, return default port (will trigger auto-start)
    print(f"No ChimeraX instances found, will try to start on port {_default_port}", file=sys.stderr)
    return _default_port

async def run_chimerax_command(command: str, port: int = None) -> str:
    """Execute a ChimeraX command via REST API on specified instance"""

    if port is None:
        # Auto-discover the best instance to use
        port = await find_best_chimerax_instance()

    session = await get_session()
    base_url = get_chimerax_url(port)

    try:
        # Use GET with query parameters - more reliable for single commands
        url = f"{base_url}/run"
        params = {'command': command}

        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.text()
            else:
                error_text = await response.text()
                raise Exception(f"ChimeraX returned status {response.status}: {error_text}")

    except aiohttp.ClientConnectorError:
        # Try to start ChimeraX if it's not running
        success, actual_port = await start_chimerax(port)
        if success:
            # Update URL if port changed
            if actual_port != port:
                base_url = get_chimerax_url(actual_port)
                url = f"{base_url}/run"

            # Retry the command after starting ChimeraX
            params = {'command': command}
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    error_text = await response.text()
                    raise Exception(f"ChimeraX returned status {response.status}: {error_text}")
        else:
            raise Exception(f"Cannot connect to ChimeraX at {base_url} and failed to start ChimeraX automatically.")
    except Exception as e:
        raise Exception(f"Error communicating with ChimeraX: {e}")

# Tool definitions using FastMCP decorators

@mcp.tool()
async def run_command(command: str, session_id: int = None) -> str:
    """Execute any ChimeraX command directly. USE THIS TOOL FIRST for any ChimeraX operation.
    For command syntax help, check the documentation resource chimerax://command/<command_name>.

    Args:
        command: ChimeraX command to execute (e.g., 'open 1gcn', 'color red')
        session_id: ChimeraX session port (defaults to primary session)
    """
    result = await run_chimerax_command(command, session_id)
    session_info = f" on session {session_id}" if session_id else ""

    # Add helpful hints for common issues
    hints = ""
    if "error" in result.lower():
        cmd_name = command.split()[0] if command.split() else command
        hints = f"\n💡 For command help, check the documentation resource: chimerax://command/{cmd_name}"

    return f"Command executed{session_info}: {command}\nOutput:\n{result}{hints}"

@mcp.tool()
async def open_structure(identifier: str, format: str = "auto-detect", session_id: int = None) -> str:
    """Open a molecular structure file or fetch from PDB

    Args:
        identifier: PDB ID (e.g., '1gcn') or file path to open
        format: File format if needed (pdb, cif, etc.), defaults to auto-detect
        session_id: ChimeraX session port (defaults to primary session)
    """
    if format != "auto-detect":
        command = f"open {identifier} format {format}"
    else:
        command = f"open {identifier}"

    result = await run_chimerax_command(command, session_id)
    session_info = f" in session {session_id}" if session_id else ""
    return f"Opened structure: {identifier}{session_info}\nOutput:\n{result}"

@mcp.tool()
async def list_models(session_id: int = None) -> str:
    """List all models currently loaded in ChimeraX

    Args:
        session_id: ChimeraX session port (defaults to primary session)
    """
    result = await run_chimerax_command("info models", session_id)
    session_info = f" in session {session_id}" if session_id else ""
    return f"Models{session_info}:\n{result}"

@mcp.tool()
async def get_model_info(model_id: str, session_id: int = None) -> str:
    """Get detailed information about a specific model

    Args:
        model_id: Model ID (e.g., '#1' or '#1.1')
        session_id: ChimeraX session port (defaults to primary session)
    """
    result = await run_chimerax_command(f"info model #{model_id}", session_id)
    session_info = f" in session {session_id}" if session_id else ""
    return f"Model #{model_id} information{session_info}:\n{result}"

@mcp.tool()
async def show_hide_models(model_spec: str, action: str, session_id: int = None) -> str:
    """Show or hide models in the display

    Args:
        model_spec: Model specification (e.g., '#1', '#2', 'all')
        action: Whether to show or hide the specified models ('show' or 'hide')
        session_id: ChimeraX session port (defaults to primary session)
    """
    if action not in ["show", "hide"]:
        raise ValueError("Action must be 'show' or 'hide'")

    command = f"{action} {model_spec}"
    result = await run_chimerax_command(command, session_id)
    session_info = f" in session {session_id}" if session_id else ""
    return f"Action '{action}' applied to {model_spec}{session_info}\nOutput:\n{result}"

@mcp.tool()
async def color_models(color: str, target: str = "all", session_id: int = None) -> str:
    """Color models or parts of models

    Args:
        color: Color name or hex code (e.g., 'red', 'blue', '#ff0000')
        target: What to color (e.g., '#1', 'protein', 'ligand'), defaults to 'all'
        session_id: ChimeraX session port (defaults to primary session)
    """
    command = f"color {target} {color}"
    result = await run_chimerax_command(command, session_id)
    session_info = f" in session {session_id}" if session_id else ""
    return f"Colored {target} with {color}{session_info}\nOutput:\n{result}"

@mcp.tool()
async def save_image(filename: str, width: int = 1920, height: int = 1080, supersample: int = 3, session_id: int = None) -> str:
    """Save a screenshot of the current view

    Args:
        filename: Output filename (e.g., 'structure.png')
        width: Image width in pixels (default: 1920)
        height: Image height in pixels (default: 1080)
        supersample: Supersampling factor for higher quality (default: 3)
        session_id: ChimeraX session port (defaults to primary session)
    """
    command = f"save {filename} width {width} height {height} supersample {supersample}"
    result = await run_chimerax_command(command, session_id)
    session_info = f" from session {session_id}" if session_id else ""
    return f"Saved image: {filename}{session_info} ({width}x{height}, supersample {supersample})\nOutput:\n{result}"

@mcp.tool()
async def capture_view(width: int = 800, height: int = 600, supersample: int = 1, session_id: int = None) -> str:
    """Capture the current ChimeraX graphics view and return it as base64-encoded image data

    Args:
        width: Image width in pixels (default: 800)
        height: Image height in pixels (default: 600)
        supersample: Supersampling factor for higher quality (default: 1)
        session_id: ChimeraX session port (defaults to primary session)
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        # Save the current view to the temporary file
        command = f"save {temp_path} width {width} height {height} supersample {supersample}"
        result = await run_chimerax_command(command, session_id)

        # Read the image file and encode as base64
        with open(temp_path, "rb") as img_file:
            img_data = img_file.read()
            base64_data = base64.b64encode(img_data).decode('utf-8')

        return f"data:image/png;base64,{base64_data}"

    except Exception as e:
        return f"Error capturing view: {e}"

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

@mcp.tool()
async def session_info(session_id: int = None) -> str:
    """Get general information about the current ChimeraX session

    Args:
        session_id: ChimeraX session port (defaults to primary session)
    """
    result = await run_chimerax_command("info", session_id)
    session_info = f" for session {session_id}" if session_id else ""
    return f"ChimeraX session information{session_info}:\n{result}"

# Instance management tools

@mcp.tool()
async def list_chimerax_instances() -> str:
    """List all running ChimeraX instances"""
    instances = await list_running_instances()
    if not instances:
        return "No ChimeraX instances are currently running."

    result = "Running ChimeraX instances:\n"
    for port, info in instances.items():
        status = info.get("status", "unknown")
        session_name = info.get("session_name", f"session_{port}")
        started_by = info.get("started_by", "unknown")
        result += f"  Port {port}: {session_name} (status: {status}, started by: {started_by})\n"

    return result

@mcp.tool()
async def start_new_chimerax_session(session_name: str = None, port: int = None) -> str:
    """Start a new ChimeraX instance/session

    Args:
        session_name: Optional name for the session (defaults to session_<port>)
        port: Specific port to use (defaults to finding available port)
    """
    if port is None:
        port = find_available_port()

    # Check if already running on this port
    if await is_chimerax_running(port):
        # If a specific session name was requested, still update the instance info
        if session_name and port in _instances:
            _instances[port]["session_name"] = session_name
            return f"Using existing ChimeraX session (renamed to: {session_name}) on port {port}"

        # Suggest an available port for a new session
        available_port = find_available_port(port + 1)
        existing_session = _instances.get(port, {}).get("session_name", f"session_{port}")
        return f"ChimeraX is already running on port {port} (session: {existing_session}). To start a new session, use port {available_port} instead."

    # Force a new instance by not reusing existing ones when session_name is provided
    success, actual_port = await start_chimerax(port, session_name, force_new=bool(session_name))
    if success:
        # Use the requested session name, or get from instances if not provided
        if session_name is None:
            instance_info = _instances.get(actual_port, {})
            session_name = instance_info.get("session_name", f"session_{actual_port}")
        else:
            # Update instances with the correct session name if we started new one
            if actual_port in _instances:
                _instances[actual_port]["session_name"] = session_name
        return f"Started new ChimeraX session: {session_name} on port {actual_port}"
    else:
        return f"Failed to start ChimeraX session on port {port}"

@mcp.tool()
async def check_chimerax_status(session_id: int = None) -> str:
    """Check if ChimeraX is running and accessible

    Args:
        session_id: ChimeraX session port to check (defaults to primary session)
    """
    port = session_id if session_id is not None else _default_port

    if await is_chimerax_running(port):
        if port in _instances:
            info = _instances[port]
            session_name = info.get("session_name", f"session_{port}")
            return f"ChimeraX session '{session_name}' is running and accessible on port {port}."
        else:
            return f"ChimeraX is running and accessible on port {port} (discovered session)."
    else:
        chimerax_path = find_chimerax_executable()
        if chimerax_path:
            return f"ChimeraX is not running on port {port}. Found executable at: {chimerax_path}"
        else:
            return f"ChimeraX is not running on port {port} and executable not found in common locations."

@mcp.tool()
async def set_default_session(session_id: int) -> str:
    """Set the default ChimeraX session for commands without explicit session_id

    Args:
        session_id: ChimeraX session port to use as default
    """
    global _default_port

    if not await is_chimerax_running(session_id):
        return f"No ChimeraX instance running on port {session_id}"

    old_default = _default_port
    _default_port = session_id

    session_name = "unknown session"
    if session_id in _instances:
        session_name = _instances[session_id].get("session_name", f"session_{session_id}")

    return f"Default session changed from port {old_default} to port {session_id} ({session_name})"

# Documentation Resources using MCP resources

@mcp.resource("chimerax://commands")
async def list_commands() -> str:
    """List all available ChimeraX commands"""
    commands = list_available_commands()
    if not commands:
        return "No ChimeraX documentation found"

    result = "ChimeraX Available Commands:\n\n"
    for i, cmd in enumerate(commands):
        result += f"• {cmd}\n"
        if i > 100:  # Limit for readability
            result += f"... and {len(commands) - 100} more commands\n"
            break

    result += f"\nTotal: {len(commands)} commands available\n"
    result += "Use chimerax://command/<name> to get detailed documentation for any command."
    return result

@mcp.resource("chimerax://command/{command_name}")
async def get_command_documentation(command_name: str) -> str:
    """Get detailed documentation for a specific ChimeraX command"""
    return get_command_doc(command_name)

# Cleanup function for aiohttp session
async def cleanup():
    """Clean up resources"""
    global _session
    if _session and not _session.closed:
        await _session.close()

if __name__ == "__main__":
    import atexit
    atexit.register(lambda: asyncio.run(cleanup()))

    mcp.run()