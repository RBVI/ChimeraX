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

async def is_chimerax_running(port: Optional[int] = None) -> bool:
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

def get_chimerax_url(port: Optional[int] = None) -> str:
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

async def start_chimerax(port: Optional[int] = None, session_name: Optional[str] = None, force_new: bool = False) -> tuple[bool, int]:
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

async def _execute_command_request(session, url: str, command: str) -> dict:
    """Helper to execute a ChimeraX command and parse the response.
    
    Args:
        session: aiohttp ClientSession
        url: Full URL to the ChimeraX /run endpoint
        command: ChimeraX command to execute
    
    Returns:
        dict with 'return_values', 'json_values', and 'logs' keys
    
    Raises:
        Exception if request fails or ChimeraX returns an error
    """
    params = {'command': command}
    
    async with session.get(url, params=params) as response:
        if response.status == 200:
            # Parse JSON response
            data = await response.json()
            
            # Check for errors - raise exception if present
            if data.get("error") is not None:
                error_info = data["error"]
                error_type = error_info.get("type", "UnknownError")
                error_msg = error_info.get("message", "Unknown error")
                raise Exception(f"{error_type}: {error_msg}")
            
            # Extract structured data
            return {
                "return_values": data.get("python values", []),
                "json_values": data.get("json values", []),
                "logs": data.get("log messages", {})
            }
        else:
            error_text = await response.text()
            raise Exception(f"ChimeraX returned status {response.status}: {error_text}")


async def run_chimerax_command(command: str, port: Optional[int] = None) -> dict:
    """Execute a ChimeraX command via REST API on specified instance
    
    Returns a structured dict with:
        - return_values: list of Python values returned by commands
        - json_values: list of JSON values returned by commands
        - logs: dict with log messages organized by level (error, warning, info, etc.)
    
    Raises Exception if ChimeraX command returns an error.
    """

    if port is None:
        # Auto-discover the best instance to use
        port = await find_best_chimerax_instance()

    session = await get_session()
    base_url = get_chimerax_url(port)
    url = f"{base_url}/run"

    try:
        # Use GET with query parameters (JSON mode enabled at startup)
        return await _execute_command_request(session, url, command)

    except aiohttp.ClientConnectorError:
        # Try to start ChimeraX if it's not running
        success, actual_port = await start_chimerax(port)
        if success:
            # Update URL if port changed
            if actual_port != port:
                base_url = get_chimerax_url(actual_port)
                url = f"{base_url}/run"
            
            # Retry the command after starting ChimeraX
            return await _execute_command_request(session, url, command)
        else:
            raise Exception(f"Cannot connect to ChimeraX at {base_url} and failed to start ChimeraX automatically.")
    except Exception as e:
        # Re-raise if already a ChimeraX error
        if isinstance(e, Exception) and any(err_type in str(e) for err_type in ["Error:", "Exception:"]):
            raise
        raise Exception(f"Error communicating with ChimeraX: {e}")

def format_chimerax_response(result: dict, context: str = "") -> str:
    """Format structured ChimeraX response into readable string
    
    Args:
        result: Dict with 'return_values', 'json_values', and 'logs' keys
        context: Optional context string to prepend to output
    
    Returns:
        Formatted string with context and log messages organized by level
    """
    output = []
    
    # Add context if provided
    if context:
        output.append(context)
    
    # Format logs by level (in order of severity)
    logs = result.get("logs", {})
    for level in ["error", "warning", "info", "debug"]:
        messages = logs.get(level, [])
        if messages:
            # Filter out empty messages
            filtered_messages = [msg for msg in messages if msg.strip()]
            if filtered_messages:
                output.append(f"{level.upper()}: {'; '.join(filtered_messages)}")
    
    # If no output generated, indicate success
    if not output:
        return "Command completed successfully"
    
    return "\n".join(output)

# Tool definitions using FastMCP decorators

@mcp.tool()
async def run_command(command: str, session_id: Optional[int] = None) -> str:
    """Execute any ChimeraX command directly. USE THIS TOOL FIRST for any ChimeraX operation.
    For command syntax help, check the documentation resource chimerax://command/<command_name>.

    Args:
        command: ChimeraX command to execute (e.g., 'open 1gcn', 'color red')
        session_id: ChimeraX session port (defaults to primary session)
    """
    result = await run_chimerax_command(command, session_id)
    session_info = f" on session {session_id}" if session_id else ""
    context = f"Command executed{session_info}: {command}"
    
    return format_chimerax_response(result, context)

@mcp.tool()
async def open_structure(identifier: str, format: str = "auto-detect", session_id: Optional[int] = None) -> str:
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
    context = f"Opened structure: {identifier}{session_info}"
    
    return format_chimerax_response(result, context)

def _format_single_model_info(model: dict) -> list:
    """Helper function to format a single model's information into lines of text.
    
    Args:
        model: Dictionary containing model data from ChimeraX info command
        
    Returns:
        List of strings representing formatted lines for this model
    """
    output = []
    
    # Basic model line: #id, name, shown/hidden
    spec = model.get('spec', '').lstrip('#')
    name = model.get('name', 'unnamed')
    shown = model.get('shown', False)
    visibility = 'shown' if shown else 'hidden'
    
    line = f"#{spec}, {name}, {visibility}"
    
    # Add triangle count if present
    triangles = model.get('num triangles', 0)
    if triangles > 0:
        line += f", {triangles} triangles"
    
    output.append(line)
    
    # For atomic structures, add detailed info on next line
    num_atoms = model.get('num atoms')
    if num_atoms is not None and num_atoms > 0:
        details = []
        details.append(f"{num_atoms} atoms")
        
        num_bonds = model.get('num bonds')
        if num_bonds is not None:
            details.append(f"{num_bonds} bonds")
        
        num_residues = model.get('num residues')
        if num_residues is not None:
            details.append(f"{num_residues} residues")
        
        chains = model.get('chains', [])
        if chains:
            chain_list = ','.join(chains)
            details.append(f"{len(chains)} chains ({chain_list})")
        
        output.append(', '.join(details))
        
        # Add pseudobond group info if present
        pbg = model.get('pseudobond groups', [])
        for pg in pbg:
            pg_name = pg.get('name', 'unknown')
            pg_count = pg.get('num pseudobonds', 0)
            output.append(f"{pg_count} {pg_name}")
    
    # For pseudobond groups, add pseudobond count
    num_pseudobonds = model.get('num pseudobonds')
    if num_pseudobonds is not None and num_atoms is None:
        # Only show if it's a standalone pseudobond group (not part of structure)
        pass  # Already included in main line via triangles or can be added here
    
    # For volumes, add volume-specific info
    size = model.get('size')
    if size is not None:
        vol_details = []
        vol_details.append(f"size {','.join(map(str, size))}")
        
        step = model.get('step')
        if step is not None:
            vol_details.append(f"step {step}")
        
        voxel_size = model.get('voxel size')
        if voxel_size is not None:
            vol_details.append(f"voxel size {voxel_size}")
        
        # Add level info
        surface_levels = model.get('surface levels', [])
        if surface_levels:
            levels_str = ', '.join(str(l) for l in surface_levels)
            vol_details.append(f"level {levels_str}")
        
        # Add value range
        min_val = model.get('minimum value')
        max_val = model.get('maximum value')
        if min_val is not None and max_val is not None:
            vol_details.append(f"value range {min_val} - {max_val}")
        
        value_type = model.get('value type')
        if value_type is not None:
            vol_details.append(f"value type {value_type}")
        
        num_sym = model.get('num symmetry operators', 0)
        vol_details.append(f"{num_sym} symmetry operators")
        
        # Replace the main line with volume info
        output[-1] = f"#{spec}, {name}, {visibility} " + ', '.join(vol_details)
    
    return output

@mcp.tool()
async def list_models(session_id: Optional[int] = None) -> str:
    """List all models currently loaded in ChimeraX with key details.

    Use this regularly to check whether the expected models are visible.
    
    Returns a summary line with the total number of models, followed by detailed 
    information for each model, including whether it is visible. 
    The format varies by model type:
    
    - Model ID (e.g., #1, #1.1, #2)
    - Model name
    - Visibility status (shown/hidden)
    
    For AtomicStructure models:
        - Number of atoms
        - Number of bonds
        - Number of residues
        - Number of chains with chain IDs (e.g., "4 chains (D,A,C,B)")
        - Missing structure pseudobond groups (if any)
    
    For PseudobondGroup models:
        - Number of pseudobonds
    
    For Volume models:
        - Size (e.g., "size 400,400,400")
        - Step value
        - Voxel size
        - Level value(s)
        - Value range (min - max)
        - Value type (e.g., float32, int16)
        - Number of symmetry operators
    
    For Surface models:
        - Number of triangles
    
    For ObjectLabels models:
        - Number of triangles
    
    Example output:
        Models:
        INFO: 3 models
        #1, 7msa, shown
        7200 atoms, 7240 bonds, 1006 residues, 4 chains (D,A,C,B)
        11 missing structure
        #1.1, missing structure, shown, 11 pseudobonds
        #1.1.1, labels, shown, 22 triangles

    Args:
        session_id: ChimeraX session port (defaults to primary session)
    """
    result = await run_chimerax_command("info", session_id)
    session_info = f" in session {session_id}" if session_id else ""
    
    # The info command returns JSON data when the REST server is in JSON mode
    json_values = result.get("json_values", [])
    
    if json_values and len(json_values) > 0:
        # We got JSON data - need to format it back into the text representation
        import json
        model_data = json_values[0] if isinstance(json_values[0], list) else json.loads(json_values[0])
        
        output = [f"Models{session_info}:"]
        output.append(f"{len(model_data)} models")
        
        for model in model_data:
            # Use helper function to format this model
            model_lines = _format_single_model_info(model)
            output.extend(model_lines)
        
        context = "\n".join(output)
        return format_chimerax_response(result, context)
    else:
        output = [f"Models{session_info}:"]
        output.append("No models loaded")
        
        context = "\n".join(output)
        return format_chimerax_response(result, context)

@mcp.tool()
async def get_model_info(model_id: str, session_id: Optional[int] = None) -> str:
    """Get detailed information about a specific model

    Args:
        model_id: Model ID (e.g., '#1' or '#1.1')
        session_id: ChimeraX session port (defaults to primary session)
    """
    # Strip '#' from model_id if present
    model_id_clean = model_id.lstrip('#')
    
    # Call info command to get all models (reusing list_models approach)
    result = await run_chimerax_command("info", session_id)
    
    # Parse JSON output
    json_values = result.get("json_values", [])
    
    if not json_values or len(json_values) == 0:
        return f"No model information available"
    
    # Parse model data
    import json
    model_data = json_values[0] if isinstance(json_values[0], list) else json.loads(json_values[0])
    
    # Find the specific model by spec
    target_model = None
    for model in model_data:
        spec = model.get('spec', '').lstrip('#')
        if spec == model_id_clean:
            target_model = model
            break
    
    if target_model is None:
        return f"Model #{model_id_clean} not found"
    
    # Format model information using helper function
    output = _format_single_model_info(target_model)
    
    # Collect all chain results for combined log handling
    all_chain_results = []
    
    # Get chain information for each chain
    chains = target_model.get('chains', [])
    if chains:
        for chain_id in chains:
            try:
                chain_info, chain_result = await _get_chain_info_helper(model_id_clean, chain_id, session_id)
                output.append(chain_info)
                all_chain_results.append(chain_result)
            except Exception as e:
                output.append(f"Chain {chain_id}: Error retrieving information - {e}")
    
    # Combine logs from info command and all chain info commands
    combined_logs = {}
    for res in [result] + all_chain_results:
        for level, messages in res.get("logs", {}).items():
            if level not in combined_logs:
                combined_logs[level] = []
            combined_logs[level].extend(messages)
    
    combined_result = {
        "return_values": result.get("return_values", []),
        "json_values": result.get("json_values", []),
        "logs": combined_logs
    }
    
    session_info = f" in session {session_id}" if session_id else ""
    context = f"Model information for #{model_id_clean}{session_info}:\n" + "\n".join(output)
    
    return format_chimerax_response(combined_result, context)

async def _get_chain_info_helper(model_id: str, chain_id: str, session_id: Optional[int] = None) -> tuple[str, dict]:
    """Helper function to get chain information without formatting the response.
    
    Returns a tuple of (formatted_string, combined_result_dict) where combined_result_dict
    contains all the logs from the commands executed.
    
    Args:
        model_id: Model ID (e.g., '1' for #1)
        chain_id: Chain ID (e.g., 'A')
        session_id: ChimeraX session port (defaults to primary session)
    
    Returns:
        Tuple of (formatted chain info string, combined result dict with logs)
    """
    # Build the chain specification
    chain_spec = f"#{model_id}/{chain_id}"
    
    # Attributes to query
    attributes = ["chain_id", "polymer_type", "description", "num_residues", "num_existing_residues"]
    
    # Collect information from each attribute
    chain_data = {}
    all_results = []
    
    for attr in attributes:
        command = f"info chains {chain_spec} attribute {attr}"
        result = await run_chimerax_command(command, session_id)
        all_results.append(result)
        
        # Parse the output - check all log levels
        logs = result.get("logs", {})
        
        # Try all log levels (messages can be in different log levels)
        all_messages = []
        for level in logs:
            all_messages.extend(logs[level])
        
        if all_messages:
            # Expected format: "chain id /A chain_id A" or "chain id /A description Estrogen receptor"
            # We want to extract the value after the attribute name
            # Note: messages include command echo AND result, so we need to find the result line
            for msg in all_messages:
                # Skip command echo - look for lines starting with "chain id"
                msg_stripped = msg.strip()
                if not msg_stripped.startswith("chain id"):
                    continue
                
                # The message format is: "chain id /{chain_id} {attribute_name} {value}"
                # Find the attribute name and extract everything after it
                parts = msg_stripped.split()
                
                # Look for the attribute name in the parts list
                if attr in parts:
                    attr_index = parts.index(attr)
                    # Everything after the attribute name is the value
                    if attr_index + 1 < len(parts):
                        value_parts = parts[attr_index + 1:]
                        value = ' '.join(value_parts)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        chain_data[attr] = value
                        break
    
    # Map polymer_type to readable name
    polymer_type_map = {
        "1": "Protein",
        "2": "Nucleic Acid"
    }
    
    # Build the formatted output line
    chain_id_val = chain_data.get("chain_id", "Unknown")
    polymer_type = chain_data.get("polymer_type", "Unknown")
    polymer_type_str = polymer_type_map.get(polymer_type, f"Other (type {polymer_type})")
    description = chain_data.get("description", "No description")
    num_residues = chain_data.get("num_residues", "Unknown")
    num_existing = chain_data.get("num_existing_residues", "Unknown")
    
    # Format the output line
    output = (f"Chain ID = {chain_id_val} | "
             f"Type = {polymer_type_str} | "
             f"Description = {description} | "
             f"Number of residues = {num_residues} | "
             f"Num. of residues that have atomic coordinates = {num_existing}")
    
    # Combine logs from all commands
    combined_logs = {}
    for result in all_results:
        for level, messages in result.get("logs", {}).items():
            if level not in combined_logs:
                combined_logs[level] = []
            combined_logs[level].extend(messages)
    
    combined_result = {
        "return_values": sum([r.get("return_values", []) for r in all_results], []),
        "json_values": sum([r.get("json_values", []) for r in all_results], []),
        "logs": combined_logs
    }
    
    return output, combined_result

@mcp.tool()
async def get_chain_info(model_id: str, chain_id: str, session_id: Optional[int] = None) -> str:
    """Get detailed information about a specific chain in a model
    
    Returns a formatted summary line with:
    - Chain ID
    - Type (Protein, Nucleic Acid, or Other)
    - Description
    - Total number of residues
    - Number of residues with atomic coordinates

    Args:
        model_id: Model ID (e.g., '1' for #1)
        chain_id: Chain ID (e.g., 'A')
        session_id: ChimeraX session port (defaults to primary session)
    """
    chain_info, result = await _get_chain_info_helper(model_id, chain_id, session_id)
    session_info = f" in session {session_id}" if session_id else ""
    context = f"Chain information for #{model_id}/{chain_id}{session_info}:\n{chain_info}"
    
    return format_chimerax_response(result, context)

@mcp.tool()
async def show_hide_models(model_spec: str, action: str, session_id: Optional[int] = None) -> str:
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
    context = f"Action '{action}' applied to {model_spec}{session_info}"
    
    return format_chimerax_response(result, context)

@mcp.tool()
async def color_models(color: str, target: str = "all", session_id: Optional[int] = None) -> str:
    """Color models or parts of models

    Args:
        color: Color name or hex code (e.g., 'red', 'blue', '#ff0000')
        target: What to color (e.g., '#1', 'protein', 'ligand'), defaults to 'all'
        session_id: ChimeraX session port (defaults to primary session)
    """
    command = f"color {target} {color}"
    result = await run_chimerax_command(command, session_id)
    session_info = f" in session {session_id}" if session_id else ""
    context = f"Colored {target} with {color}{session_info}"
    
    return format_chimerax_response(result, context)

@mcp.tool()
async def save_image(filename: str, width: int = 1920, height: int = 1080, supersample: int = 3, session_id: Optional[int] = None) -> str:
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
    context = f"Saved image: {filename}{session_info} ({width}x{height}, supersample {supersample})"
    
    return format_chimerax_response(result, context)


@mcp.tool()
async def superpose_residue(
    source_model: str,
    source_chain: str,
    source_residue: str,
    target_model: str,
    target_chain: str,
    target_residue: str,
    session_id: Optional[int] = None
) -> str:
    """Move a residue from one model to superpose with a residue in another model.
    
    This is useful when the 'align' command won't work because the two residues 
    don't have the same atoms (e.g., different small molecule ligands).
    
    The tool works by:
    1. Centering the view on the target residue (sets center of rotation)
    2. Moving the source residue to the center of rotation
    
    Args:
        source_model: Model number containing the residue to move (e.g., '2')
        source_chain: Chain ID of the residue to move (e.g., 'A')
        source_residue: Residue name/number to move (e.g., 'DRG' or '100')
        target_model: Model number containing the target residue (e.g., '1')
        target_chain: Chain ID of the target residue (e.g., 'A')
        target_residue: Residue name/number to align to (e.g., 'DRG' or '100')
        session_id: ChimeraX session port (defaults to primary session)
    """
    # Build selection strings
    target_spec = f"#{target_model}/{target_chain}:{target_residue}"
    source_spec = f"#{source_model}/{source_chain}:{source_residue}"
    
    session_info = f" in session {session_id}" if session_id else ""
    
    # Step 1: Center view on target residue (sets center of rotation)
    view_command = f"view {target_spec}"
    view_result = await run_chimerax_command(view_command, session_id)
    
    # Step 2: Move source residue to center of rotation
    move_command = f"move cofr {source_spec}"
    move_result = await run_chimerax_command(move_command, session_id)
    
    # Format combined results
    context = (f"Successfully superposed residue{session_info}:\n"
               f"  Source: {source_spec}\n"
               f"  Target: {target_spec}")
    
    # Combine logs from both commands
    combined_logs = {}
    for result in [view_result, move_result]:
        for level, messages in result.get("logs", {}).items():
            if level not in combined_logs:
                combined_logs[level] = []
            combined_logs[level].extend(messages)
    
    combined_result = {
        "return_values": view_result.get("return_values", []) + move_result.get("return_values", []),
        "json_values": view_result.get("json_values", []) + move_result.get("json_values", []),
        "logs": combined_logs
    }
    
    return format_chimerax_response(combined_result, context)

@mcp.tool()
async def show_hide_hydrogens(
    action: str,
    hydrogen_type: str = "all",
    target: str = "",
    session_id: Optional[int] = None
) -> str:
    """Show or hide hydrogen atoms (all, polar only, or nonpolar only)
    
    In ChimeraX:
    - 'H' refers to all hydrogen atoms
    - 'HC' refers to nonpolar hydrogens (hydrogens bonded to carbon)
    - Polar hydrogens are H atoms that are not HC
    
    Args:
        action: 'show' or 'hide'
        hydrogen_type: Type of hydrogens - 'all', 'polar', or 'nonpolar' (default: 'all')
        target: Optional target specification to limit scope (e.g., '#1', ':ALA', default: all models)
        session_id: ChimeraX session port (defaults to primary session)
    
    Examples:
        - Show all hydrogens: action='show', hydrogen_type='all'
        - Hide all hydrogens: action='hide', hydrogen_type='all'
        - Show only polar hydrogens: action='show', hydrogen_type='polar'
        - Hide nonpolar hydrogens: action='hide', hydrogen_type='nonpolar'
    """
    if action not in ["show", "hide"]:
        raise ValueError("Action must be 'show' or 'hide'")
    
    if hydrogen_type not in ["all", "polar", "nonpolar"]:
        raise ValueError("hydrogen_type must be 'all', 'polar', or 'nonpolar'")
    
    session_info = f" in session {session_id}" if session_id else ""
    commands = []
    
    # Build target specification
    target_spec = f" {target}" if target else ""
    
    if hydrogen_type == "all":
        # Simple case: show or hide all hydrogens
        command = f"{action} H{target_spec}"
        commands.append(command)
    
    elif hydrogen_type == "polar":
        if action == "show":
            # Show all H, then hide HC (nonpolar)
            commands.append(f"show H{target_spec}")
            commands.append(f"hide HC{target_spec}")
        else:  # hide
            # Hide H but not HC (using negation)
            command = f"hide H{target_spec} & ~HC{target_spec}"
            commands.append(command)
    
    elif hydrogen_type == "nonpolar":
        # Nonpolar hydrogens are HC
        command = f"{action} HC{target_spec}"
        commands.append(command)
    
    # Execute commands and collect results
    results = []
    for cmd in commands:
        result = await run_chimerax_command(cmd, session_id)
        results.append(result)
    
    # Format combined results
    context = (f"Successfully {action} {hydrogen_type} hydrogens{session_info}\n"
               f"Target: {target if target else 'all models'}")
    
    # Combine logs from all commands
    combined_logs = {}
    for result in results:
        for level, messages in result.get("logs", {}).items():
            if level not in combined_logs:
                combined_logs[level] = []
            combined_logs[level].extend(messages)
    
    combined_result = {
        "return_values": sum([r.get("return_values", []) for r in results], []),
        "json_values": sum([r.get("json_values", []) for r in results], []),
        "logs": combined_logs
    }
    
    return format_chimerax_response(combined_result, context)

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
async def start_new_chimerax_session(session_name: Optional[str] = None, port: Optional[int] = None) -> str:
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
async def check_chimerax_status(session_id: Optional[int] = None) -> str:
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