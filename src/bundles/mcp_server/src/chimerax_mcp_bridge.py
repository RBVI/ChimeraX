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

Session Management Behavior:
1. Default port starts at 8080 (configurable)
2. start_new_chimerax_session() ALWAYS forces a new instance (no auto-reuse)
3. Commands without session_id use auto-discovery:
   - First checks default port
   - Then scans common ports [8081, 8082, 8083, 7955, 9000]
   - If none found, auto-starts on default port
4. Auto-discovery NO LONGER changes the default port (use set_default_session() explicitly)
5. check_chimerax_status() only checks, never starts ChimeraX
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

# Debug settings
DEBUG = False  # Set to True to enable debug logging to /tmp/

# Global state for managing multiple ChimeraX instances
_instances = {}  # port -> instance info
_default_port = DEFAULT_CHIMERAX_PORT

from mcp.server.fastmcp import FastMCP

def find_chimerax_executable():
    """Find ChimeraX executable, starting from bridge script location"""

    install_dir = _chimerax_installation_directory()
    if install_dir:
        from sys import platform
        if platform == 'darwin':
            exe_path = os.path.join(install_dir, 'Contents', 'MacOS', 'ChimeraX')
        elif platform == 'win32':
            exe_path = os.path.join(install_dir, 'bin', 'ChimeraX.exe')
        else:
            exe_path = os.path.join(install_dir, 'bin', 'ChimeraX')  # Linux
        if not os.path.exists(exe_path):
            exe_path = None
    else:
        exe_path = None

    return exe_path

def _chimerax_installation_directory():
    # First, try to find ChimeraX relative to this bridge script
    bridge_path = os.path.abspath(__file__)

    from sys import platform
    if platform == 'darwin':
        cdir = _find_parent_directory(bridge_path, 'Contents')
    elif platform == 'win32':
        cdir = _find_parent_directory(bridge_path, 'bin')
    else:
        # Linux
        cdir = _find_parent_directory(bridge_path, 'lib')

    install_dir = os.path.dirname(cdir) if cdir else None

    return install_dir
    
def _find_parent_directory(path, dir_name):
    while path:
        if os.path.basename(path) == dir_name:
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
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
            # First child exits immediately without Python cleanup to avoid fork crash
            os._exit(0)

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
    """Start ChimeraX if not running, returns (success, port)
    
    Args:
        port: Port to start ChimeraX on (defaults to _default_port)
        session_name: Optional name for the session
        force_new: If True, don't try to reuse existing instances
    """
    if port is None:
        port = _default_port

    # First check if there's already a ChimeraX with REST server running (unless forcing new)
    # NOTE: We only do this auto-discovery if port is None and force_new is False
    # This prevents unexpected reuse when a specific port is requested
    if not force_new and port == _default_port:
        has_rest, existing_port = await check_existing_rest_server()
        if has_rest:
            print(f"Using existing ChimeraX REST server on port {existing_port} (default port: {_default_port})", file=sys.stderr)
            return True, existing_port

    # Check if already running on this port
    if await is_chimerax_running(port):
        print(f"ChimeraX is already running on port {port}", file=sys.stderr)
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

    install_dir = _chimerax_installation_directory()
    if install_dir:
        from sys import platform
        if platform == 'darwin':
            docs_path = os.path.join(install_dir, 'Contents', 'share', 'docs')
        elif platform == 'win32':
            docs_path = os.path.join(install_dir, 'bin', 'share', 'docs')
        else:
            docs_path = os.path.join(install_dir, 'share', 'docs')  # Linux
        if not os.path.exists(docs_path):
            docs_path = None
    else:
        docs_path = None

    return docs_path

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
    """Find the best ChimeraX instance to use (prefer existing, fallback to default port)
    
    NOTE: This function no longer mutates _default_port to avoid unexpected behavior.
    Use set_default_session() explicitly if you want to change the default port.
    """
    global _default_port

    # First check if default port is running
    if await is_chimerax_running(_default_port):
        print(f"Using default ChimeraX instance on port {_default_port}", file=sys.stderr)
        return _default_port

    # Quick scan for any running ChimeraX instances (common ports only)
    # NOTE: We scan but DO NOT change _default_port anymore to avoid confusion
    common_ports = [8081, 8082, 8083, 7955, 9000]  # Common alternatives
    for port in common_ports:
        if port == _default_port:
            continue  # Already checked

        if await is_chimerax_running(port):
            # Found one! But DON'T change the default - just use it for this operation
            print(f"WARNING: Found ChimeraX instance on port {port}, but default is still {_default_port}. "
                  f"Use set_default_session({port}) if you want to make this the default.", file=sys.stderr)
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
        Exception if request fails or ChimeraX returns an error (with helpful hints)
    """
    params = {'command': command}
    
    async with session.get(url, params=params) as response:
        if response.status == 200:
            # Parse JSON response
            data = await response.json()
            
            # DEBUG: Write full response to file
            if DEBUG:
                import json
                import time
                debug_file = f"/tmp/chimerax_response_{int(time.time() * 1000)}.json"
                try:
                    with open(debug_file, 'w') as f:
                        json.dump({
                            "command": command,
                            "url": url,
                            "raw_response": data
                        }, f, indent=2)
                    print(f"DEBUG: Full response written to {debug_file}", file=sys.stderr)
                except Exception as e:
                    print(f"DEBUG: Failed to write debug file: {e}", file=sys.stderr)
            
            # Check for errors - raise exception if present
            if data.get("error") is not None:
                error_info = data["error"]
                error_type = error_info.get("type", "UnknownError")
                error_msg = error_info.get("message", "Unknown error")
                
                # Add helpful hints to guide agents
                enhanced_error = add_error_hints(error_type, error_msg, command)
                raise Exception(enhanced_error)
            
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
    
    Implements a cascading fallback strategy:
    1. Priority 1: Format and return log messages
    2. Priority 2: If logs are empty, format and return json values
    3. Priority 3: If json values are also empty, format and return python values
    
    Args:
        result: Dict with 'return_values', 'json_values', and 'logs' keys
        context: Optional context string to prepend to output
    
    Returns:
        Formatted string with context and log messages/return values organized by level
    """
    import json
    
    output = []
    
    # Add context if provided
    if context:
        output.append(context)
    
    # Priority 1: Format logs by level (in order of severity)
    logs = result.get("logs", {})
    has_log_content = False
    for level in ["error", "warning", "info", "note", "debug"]:
        messages = logs.get(level, [])
        if messages:
            # Filter out empty messages and markdown-heavy command echoes
            filtered_messages = []
            for msg in messages:
                if msg.strip():
                    # Skip messages that are primarily markdown links (command echoes)
                    # These typically start with markdown link syntax and contain multiple links
                    msg_stripped = msg.strip()
                    if not (msg_stripped.startswith('[') and msg_stripped.count('](') >= 2):
                        filtered_messages.append(msg)
            if filtered_messages:
                output.append(f"{level.upper()}: {'; '.join(filtered_messages)}")
                has_log_content = True
    
    # Priority 2: If no log content, try json values
    if not has_log_content:
        json_values = result.get("json_values", [])
        # Filter out None/null values
        json_values = [v for v in json_values if v is not None]
        if json_values:
            output.append("\nJSON Output:")
            for i, val in enumerate(json_values):
                if len(json_values) > 1:
                    output.append(f"[Result {i+1}]")
                # Format JSON with indentation for readability
                try:
                    # If val is already a JSON string, parse it first
                    if isinstance(val, str):
                        val = json.loads(val)
                    formatted_json = json.dumps(val, indent=2, ensure_ascii=False)
                    output.append(formatted_json)
                except (TypeError, ValueError, json.JSONDecodeError):
                    # If JSON serialization/parsing fails, fall back to string representation
                    output.append(str(val))
        else:
            # Priority 3: If no json values, try python values
            python_values = result.get("return_values", [])
            # Filter out None values
            python_values = [v for v in python_values if v is not None]
            if python_values:
                output.append("\nOutput:")
                for i, val in enumerate(python_values):
                    if len(python_values) > 1:
                        output.append(f"[Result {i+1}]")
                    output.append(str(val))
    
    # If no output generated at all, indicate success
    if len(output) <= 1 and context:  # Only context was added
        output.append("Command completed successfully")
    elif not output:  # Nothing at all
        return "Command completed successfully"
    
    return "\n".join(output)

def add_error_hints(error_type: str, error_msg: str, command: str) -> str:
    """Add contextual hints to ChimeraX error messages to guide agents.
    
    Analyzes error messages and appends helpful hints that direct agents to:
    - The specify_objects prompt for atomspec syntax errors
    - list_models() tool for model-related errors
    - Documentation resources for command errors
    
    Args:
        error_type: Type of error (e.g., "UserError", "SyntaxError")
        error_msg: Original error message from ChimeraX
        command: The command that failed (for context)
    
    Returns:
        Enhanced error message with helpful hints appended
    """
    # Build the base error message
    full_error = f"{error_type}: {error_msg}"
    
    # Convert to lowercase for case-insensitive matching
    error_lower = error_msg.lower()
    
    # Pattern matching for different error categories
    hints = []
    
    # ===== Object Specification Errors =====
    # Real ChimeraX errors: "Expected an objects specifier", "invalid an atom specifier",
    # "not an atom specifier", "empty atom specifier", '"X" is not a selector name'
    # Also "Expected a keyword" from atomspec-taking commands like select, color, show, hide
    atomspec_commands = ['select', 'color', 'show', 'hide', 'style', 'cartoon', 
                         'display', 'label', 'size', 'view', 'zone', 'surface']
    cmd_first_word = command.strip().split()[0].lower() if command.strip() else ""
    
    is_atomspec_error = any(pattern in error_lower for pattern in [
        "expected an objects specifier",
        "expected a model specifier",
        "expected an atom specifier",
        "invalid an atom specifier",
        "not an atom specifier",
        "empty atom specifier",
        "is not a selector name",
        "only initial part"  # "only initial part 'X' of atom specifier valid"
    ])
    
    # Also catch "Expected a keyword" or similar when used with atomspec-taking commands
    is_atomspec_keyword_error = (
        ("expected a keyword" in error_lower or "expected keyword" in error_lower) 
        and cmd_first_word in atomspec_commands
    )
    
    if is_atomspec_error or is_atomspec_keyword_error:
        hints.append("\n\n🔍 HINT: This error indicates incorrect object specification (atomspec) syntax.")
        hints.append("→ Use the get_atomspec_guide() tool to learn the complete atomspec syntax")
        hints.append("→ Common patterns:")
        hints.append("  • #1          (entire model 1)")
        hints.append("  • #1/A        (chain A in model 1)")
        hints.append("  • #1/A:100    (residue 100 in chain A)")
        hints.append("  • @ca         (all CA atoms)")
        hints.append("  • protein     (all protein atoms)")
        hints.append("→ Model IDs MUST include # prefix: use '#1' not '1'")
    
    # ===== No Atoms Matched (less severe spec error) =====
    elif "no atoms matched" in error_lower or "nothing specified" in error_lower:
        hints.append("\n\n🔍 HINT: Your specification syntax may be correct, but no atoms matched.")
        hints.append("→ Use list_models() to see what structures/chains are loaded")
        hints.append("→ Check chain IDs are correct (case-sensitive)")
        hints.append("→ Verify residue numbers/names exist in the structure")
        hints.append("→ Use get_atomspec_guide() tool if you need help with atomspec syntax")
    
    # ===== Model/Structure Errors =====
    # Real ChimeraX errors: "No models specified by", "No atomic structures open/specified",
    # "Must specify 1 model, got X"
    elif any(pattern in error_lower for pattern in [
        "no models",
        "no atomic structures",
        "must specify 1 model",
        "must specify 1 atomic structure",
        "must specify exactly one"
    ]):
        hints.append("\n\n🔍 HINT: No models are loaded or the specified model doesn't exist.")
        hints.append("→ Use list_models() to see currently loaded structures")
        hints.append("→ Use open_structure() or run_command('open <pdb_id>') to load a structure")
        hints.append("→ Verify model IDs with # prefix (e.g., #1, #2)")
    
    # ===== Command Errors =====
    # Real ChimeraX error: "Unknown command: X" from cli.py
    elif any(pattern in error_lower for pattern in [
        "unknown command",
        "no command"
    ]):
        # Try to extract command name for more specific help
        cmd_parts = command.strip().split()
        cmd_name = cmd_parts[0] if cmd_parts else "unknown"
        
        hints.append(f"\n\n🔍 HINT: Command '{cmd_name}' is not recognized.")
        hints.append("→ Check command spelling and capitalization")
        hints.append("→ Use the list_chimerax_commands() tool to see all available commands")
        hints.append(f"→ Use get_command_documentation('{cmd_name}') tool for specific command help")
        hints.append("→ Common commands: open, color, show, hide, save, view, align")
    
    # ===== Argument/Syntax Errors =====
    # Real ChimeraX errors: "Missing or invalid X argument", "Expected X", "Should be X"
    elif any(pattern in error_lower for pattern in [
        "missing or invalid",
        "missing required",
        "expected",  # "Expected true or false", "Expected X"
        "should be",  # From Enum parsing
        "require"   # "Require 1 surface, got X"
    ]):
        # Extract command name
        cmd_parts = command.strip().split()
        cmd_name = cmd_parts[0] if cmd_parts else "unknown"
        
        hints.append(f"\n\n🔍 HINT: The '{cmd_name}' command has incorrect arguments.")
        hints.append(f"→ Use get_command_documentation('{cmd_name}') tool for correct syntax")
        hints.append("→ Check that you've included all required arguments")
        hints.append("→ Verify keyword spelling and order")
    
    # ===== File/Path Errors =====
    elif any(pattern in error_lower for pattern in [
        "cannot open",
        "file not found",
        "no such file",
        "cannot read",
        "does not exist"
    ]):
        hints.append("\n\n🔍 HINT: File or path error.")
        hints.append("→ Verify the file path is correct and the file exists")
        hints.append("→ Use absolute paths when possible")
        hints.append("→ For PDB files, try using PDB ID: open_structure('1gcn')")
    
    # ===== Generic Error (no specific pattern matched) =====
    else:
        # Only add a generic hint if we didn't match anything specific
        hints.append("\n\n🔍 HINT: ChimeraX command failed.")
        hints.append("→ Use list_models() to verify what structures are loaded")
        hints.append("→ Use get_atomspec_guide() tool for help with atom spec syntax")
        hints.append("→ Use get_command_documentation() tool for command syntax help")
    
    # Append all hints to the error message
    if hints:
        full_error += "".join(hints)
    
    return full_error

@mcp.tool()
async def get_atomspec_guide() -> str:
    """Get the complete guide for ChimeraX atomspec (object specification) syntax.
    
    Use this tool whenever you need to specify objects in ChimeraX commands, such as:
    - Selecting specific models, chains, residues, or atoms
    - Creating distance-based selections (zones)
    - Combining selections with logical operators
    - Using built-in classifications (protein, ligand, helix, etc.)
    - Querying by attributes
    
    This comprehensive guide covers all atomspec syntax patterns and common use cases.
    Always consult this when constructing object specifications for commands like:
    color, show, hide, select, style, view, and any command that acts on objects.
    """
    return """
# ChimeraX Object Specification Guide

## Overview
Most ChimeraX commands require specifying which items they should affect. This guide covers the atomspec syntax.

## Hierarchical Specifiers

The four main levels in descending order:

| Symbol | Level    | Description                                           | Example           |
|--------|----------|-------------------------------------------------------|-------------------|
| #      | Model    | Model number (hierarchical: N, N.N, N.N.N, etc.)     | #1, #1.3          |
| /      | Chain    | Chain identifier (case-insensitive unless mixed case) | /A, /B            |
| :      | Residue  | Residue number OR residue name (case-insensitive)     | :51, :glu, :asp   |
| @      | Atom     | Atom name (case-insensitive)                          | @ca, @n, @sg      |

**Key Rules:**
- Omitting a level means "all" at that level (e.g., `#1` = all atoms in model 1)
- Specifying atoms also includes bonds between them (unless you start with `=`)
- Use `#!N` to specify parent model only without submodels

## Lists and Ranges

**Numeric ranges and lists:**
- Comma-separated lists: `#1,2,5` or `:10,15,20`
- Ranges with hyphens: `:10-20` or `#1-3`
- Use `start` or `end` keywords: `:start-40`, `#1.2-end`
- Asterisk `*` as wildcard: `#*` (all models)

**Examples:**
```
#1,2:50,70-85@ca          # CA atoms in residues 50, 70-85 of models 1 and 2
/A-D,F                    # Chains A, B, C, D, and F
:lys,arg@cb               # CB atoms in lysine and arginine residues
```

## Implicit Operations

When repeating or returning to a higher level, the hierarchy resets:

```
:12:14@ca                 # All atoms of residue 12, CA atom of residue 14
/A/B:12-20@ca             # All atoms of chain A, CA atoms of residues 12-20 in chain B
/a:10-20,26/b:12-22,29@n,ca,c,o  # All atoms of chain A residues 10-20,26 plus 
                                  # N,CA,C,O atoms of chain B residues 12-22,29
```

## Built-in Classifications

**Structural:**
- `protein`, `nucleic`, `solvent`, `ligand`, `ions`
- `helix`, `strand`, `coil`
- `backbone`, `sidechain`, `sideonly`
- `main`, `ligand`, `solvent`

**Chemical:**
- Element symbols: `C`, `N`, `O`, `S`, `P`, etc.
- `H` (all hydrogens), `HC` (nonpolar H bonded to C)
- Functional groups: `aromatic`, `aromatic-ring`, `carboxylate`, `disulfide`
- Atom types: `C4`, `N3+`, `Car` (aromatic carbon), `Npl` (planar N), etc.

**Special:**
- `sel` - current selection
- `displayed` - currently displayed atoms

**Examples:**
```
protein & helix           # Protein atoms in helices
H & ~HC                   # Polar hydrogens (not bonded to carbon)
ligand & aromatic         # Aromatic atoms in ligands
```

## Zones (Distance-Based)

Specify atoms within or beyond a distance from a reference:

**Syntax:** `<reference> <level><operator> <distance>`

**Level symbols:** `@` (atom), `:` (residue), `/` (chain), `#` (model)
**Operators:** `<` (within/less than) or `>` (beyond/greater than)

**Examples:**
```
@nz @< 3.8                # Atoms within 3.8 Å of NZ atoms
#1:gtp :< 10.5            # Residues with any atom within 10.5 Å of GTP
(ligand | ions) @< 4.8    # Atoms within 4.8 Å of ligand or ions
(ions @< 4) & ~ions       # Atoms within 4 Å of ions, excluding ions themselves
```

## Attributes

Query by attribute values:

**Symbols:** `@@` (atom), `::` (residue), `//` (chain), `##` (model)

**Operators:** `=`, `!=`, `==` (case-sensitive), `!==`, `>`, `<`, `>=`, `<=`
**Negation:** `^` before attribute name (attribute not assigned)

**Examples:**
```
@@display                 # Displayed atoms
~@@display                # Hidden atoms
@@bfactor>40              # Atoms with B-factor > 40
@ca & @@bfactor>40        # CA atoms with B-factor > 40
::num_atoms>=10           # Residues with 10+ atoms
##name="2gbp map 5"       # Model named "2gbp map 5"
```

## Combinations

Combine specifications using operators:

- `&` - Intersection (AND) - higher priority
- `|` - Union (OR)
- `~` - Negation (NOT)
- Use parentheses `()` for grouping

**Examples:**
```
/A & protein              # Chain A protein residues only
/A & ~:hem                # Chain A except HEM residues
protein & (ligand :< 5)   # Protein residues within 5 Å of ligand
:phe,tyr & sidechain      # Phenylalanine and tyrosine sidechains
sideonly & ligand @<4     # Sidechain atoms within 4 Å of ligand
```

## Common Patterns

**Specific selections:**
```
#1/A:100-200@ca           # CA atoms of residues 100-200 in chain A of model 1
:asp,glu                  # All aspartate and glutamate residues
protein & ~backbone       # Protein sidechains only
```

**Interface analysis:**
```
#1 & (#2 :< 5)            # Model 1 residues within 5 Å of model 2
(#1 & protein) & ((#2 & ligand) :< 4)  # Protein residues near ligand
```

**Chemical groups:**
```
:cys@sg & ~disulfide      # Free cysteine sulfurs (not in disulfide bonds)
aromatic-ring & :phe,tyr  # Aromatic ring carbons in Phe and Tyr
```

## Tips

1. **Start broad, then narrow:** Begin with model/chain, then add residue/atom filters
2. **Test incrementally:** Build complex specs step by step
3. **Use sel for iteration:** Select visually, then refine with commands
4. **Blank spec = all:** An empty specification means "all applicable items"
5. **Case matters (sometimes):** Only for chain IDs when both upper/lowercase exist

## Quick Reference Card

| Task                              | Example Specification     |
|-----------------------------------|---------------------------|
| Entire model                      | `#1`                      |
| Specific chain                    | `#1/A`                    |
| Residue range                     | `#1/A:100-150`            |
| Specific atom type                | `@ca`                     |
| Multiple chains                   | `/A,B,C`                  |
| Protein backbone                  | `protein & backbone`      |
| Ligands and nearby residues       | `ligand | (ligand :< 5)`  |
| Selection within distance         | `sel @< 4`                |
| Heavy atoms (non-hydrogen)        | `~H`                      |
| Polar hydrogens                   | `H & ~HC`                 |

For more details, see: https://www.cgl.ucsf.edu/chimerax/docs/user/commands/atomspec.html
    """

@mcp.tool()
async def run_command(command: str, session_id: Optional[int] = None) -> str:
    """Execute any ChimeraX command directly. Use this tool if you don't find another tool
    that suits your needs.
    
    IMPORTANT RESTRICTIONS:
    - DO NOT use this tool for 'show' or 'hide' commands
    - For show/hide operations, use show_hide_objects() or show_hide_hydrogens() instead
    - Attempting to run 'show' or 'hide' commands will raise an exception
    
    When constructing commands that specify objects (models, chains, residues, atoms):
    - Use get_atomspec_guide() to learn the correct atomspec syntax
    - Common atomspecs: #1 (model), #1/A (chain), #1/A:100 (residue), @ca (atom type), HC (nonpolar hydrogens), #1/A & ligand (ligands in chain A of model 1)
    
    To see a list of all commands, use the list_commands() tool.
    For command syntax help, use get_command_documentation(command_name).

    Args:
        command: ChimeraX command to execute (e.g., 'open 1gcn', 'color #1/A red')
        session_id: ChimeraX session port (defaults to primary session)
    """

    '''
    # Validate that show/hide commands are not being used
    command_stripped = command.strip().lower()
    if command_stripped.startswith('show ') or command_stripped.startswith('hide ') or command_stripped == 'show' or command_stripped == 'hide':
        raise ValueError(
            "ERROR: 'show' and 'hide' commands are not allowed via run_command. "
            "Please use the dedicated show_hide_objects() or show_hide_hydrogens() tools instead. "
            "These specialized tools provide better parameter validation and error handling."
        )
    '''

    result = await run_chimerax_command(command, session_id)
    session_info = f" on session {session_id}" if session_id else ""
    context = f"Command executed{session_info}: {command}"
    
    return format_chimerax_response(result, context)

@mcp.tool()
async def open_structure(identifier: str, format: str = "auto-detect", session_id: Optional[int] = None) -> str:
    """Open a molecular structure file or fetch from PDB

    Hint:
    - After opening a structure and before showing any objects of this new model, you should first hide all its representations, so that we start from a clean slate

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
    """Get detailed information about a specific model, including details about
    all its chains. You can call this to get all chain identifications in one go.

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
async def color_models(color: str, target: str = "all", session_id: Optional[int] = None) -> str:
    """Color models or parts of models
    
    For complex target specifications, use get_atomspec_guide() to construct the correct atomspec.

    Hints:
    - "byhet" is a special color that colors non-carbon atoms by their chemical element
    - When coloring atom representations, you should always then run another color command with "byhet" as the color
    - When coloring binding pockets, you should keep the protein atoms the same color as the rest of the protein, and make the ligand a different color

    Args:
        color: Color name or hex code (e.g., 'red', 'blue', '#ff0000')
        target: What to color using atomspec syntax (e.g., '#1', '#1/A', 'protein', 'ligand'), defaults to 'all'
        session_id: ChimeraX session port (defaults to primary session)
    """
    command = f"color {target} {color}"
    result = await run_chimerax_command(command, session_id)
    session_info = f" in session {session_id}" if session_id else ""
    context = f"Colored {target} with {color}{session_info}"
    
    return format_chimerax_response(result, context)

# @mcp.tool()
async def save_image(filename: str, width: int = 1920, height: int = 1080, supersample: int = 3, session_id: Optional[int] = None) -> str:
    """Save a screenshot of the current view

    Before saving an image, clear the selection by running command "~select" - otherwise we will have bright green lights around the selected objects

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

# @mcp.tool()
async def show_hide_objects(
    action: str,
    atomspec: str,
    target: str,
    session_id: Optional[int] = None
) -> str:
    """Show or hide a specified selection of objects' representation. 
    Use this tool to show or hide specific representations (targets) of objects (atomspec) in the model, 
    rather than the generic run_command() method.
    
    For the target, use one or more of the following letter: 
    - a: atoms: Toggle on (show) or off (hide) the atomic representation of the objects specified by the atomspec
    - b: bonds: Toggle on (show) or off (hide) the bonds representation of the atomspec
    - p: pseudobonds: Toggle on (show) or off (hide) the pseudobonds
    - c: cartoons or ribbons: Toggle on (show) or off (hide) the cartoon/ribbon representation of the atomspec
    - s: surfaces: Toggle on (show) or off (hide) the surfaces associated with the atomspec    

    Examples:
        - Show all atoms and bonds in model 1: action='show', atomspec='#1', target='ab'
        - Hide all atoms and bonds and cartoons and surfaces in chain A of model 1: action='hide', atomspec='#1/A', target='abcs'
        - Show all pseudobonds in residue 100 of chain A of model 1: action='show', atomspec='#1/A:100', target='p'
        - Hide all cartoons in model 1: action='hide', atomspec='#1', target='c'
        - Hide all surfaces in model 1: action='hide', atomspec='#1', target='s'
        - Show ribbons and atoms in residues 1-50 of chain B of model 2: action='show', atomspec='#2/B:1-50', target='cb'
        - Show polar hydrogens in model 1: action='show', atomspec='#1 & H & ~HC', target='a'
        - Hide non-polar hydrogens in model 1: action='hide', atomspec='#1 & HC', target='a'
        - Show protein side chains near ligand #1/A:LIG: action='show #1 & protein & #1/A:LIG:<5', target='ab'
    
    Important:
        - Before you attempt for the first time to show a particular object, you MUST first hide all its representations (because you don't know what representations are already active)
        - After showing or hiding objects, you MUST check the response to see if the count of affected residues is what you expected. If not, you should run this tool again to correct the situation.
    
    Tips:
        - When a model is first opened by ChimeraX, it may be shown in cartoon or atoms (sticks or spheres), or some combination of the two 
        - For this reason, in general, when hiding a chain or part of the sequence, you might as well hide all the targets: abpcs
        - Also, when showing a particular representation (e.g. cartoon), you must first hide all the other representations before showing the one you want
        - If you want to show a particular ligand, specify the chain if possible. For example, prefer #1/A:LIG rather than just :LIG, which would show all the ligands named LIG
        - To make sure that only one particular representation is shown (e.g. atoms, not ribbons), you should first hide all representations before showing the one you want
        - When you want to show atoms you should  always also show the bonds, unless the user explicitly asked you to see atoms without bonds
        - When showing atoms, you should style them as sticks, not spheres; to do this, run the command "style {atomspec} stick"; also, by default you should color them by het by using the color_models tool with the color "byhet"

    Args:
        action: 'show' or 'hide'
        atomspec: Atomspec specification using atomspec syntax (e.g., '#1', '#1/A', ':ALA'; use the get_atomspec_guide() tool to see the correct syntax)
        target: What to show or hide
        session_id: ChimeraX session port (defaults to primary session)
    
    Response format:
    Success: {command}
    This action affected {counts_string}
    """
    if action not in ["show", "hide"]:
        raise ValueError("Action must be 'show' or 'hide'")

    # If target contains any letters that are not in the list of allowed letters, raise an error
    if any(letter not in ["a", "b", "p", "c", "s", "m"] for letter in target):
        raise ValueError("Target must be one or more of 'a', 'b', 'p', 'c', 's', 'm'")
    
    session_info = f" in session {session_id}" if session_id else ""

    # First, we will use the select command to select the objects specified by the atomspec
    # This has the advantage that ChimeraX will return a count of atoms, bonds, residues and models selected, which
    # is useful feedback for the client.
    # For example, the output printed to log would look something like this:
    # "108 atoms, 112 bonds, 2 residues, 1 model selected"
    select_command = f"select {atomspec}"
    select_result = await run_chimerax_command(select_command, session_id)

    # Parse the select result to get a string with the counts.
    # Note that the log level is "note" and that it will contain 2 lines: 
    # the first line is the command echo, the second line is the result which we want to grab
    counts_string = select_result.get("logs", {}).get("note", [""])[1]
    # Remove the "selected" from the counts string
    counts_string = counts_string.replace(" selected", "")

    # If the counts_string is "Nothing", raise an error
    if counts_string == "Nothing":
        raise ValueError(f"No objects were found matching the atomspec: {atomspec}")


    # The show/hide command itself
    command = f"{action} {atomspec} target {target}"
    
    # Execute the first command (if it fails, exception propagates and second command won't run)
    result = await run_chimerax_command(command, session_id)

    # If action is "show" also show the parent model
    if action == "show":
        # If we are showing a specific target (representation) of atomspec,
        # let's assume the agent means to also show the parent model.
        model_command = f"show {atomspec} target m"
        model_result = await run_chimerax_command(model_command, session_id)
        
        # Combine logs from both commands
        combined_logs = {}
        for res in [result, model_result]:
            for level, messages in res.get("logs", {}).items():
                if level not in combined_logs:
                    combined_logs[level] = []
                combined_logs[level].extend(messages)
        
        # Create combined result
        combined_result = {
            "return_values": result.get("return_values", []) + model_result.get("return_values", []),
            "json_values": result.get("json_values", []) + model_result.get("json_values", []),
            "logs": combined_logs
        }
        
        context = f"Success: {command}\nThis action affected {counts_string}"
        return format_chimerax_response(combined_result, context)
    
    # Single command case (hide or show models)
    context = f"Success: {command}"
    return format_chimerax_response(result, context)

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

    Important behavior:
    - If port is specified and ChimeraX is already running there, returns info about existing instance
    - If port is not specified, finds an available port starting from 8080
    - Always forces a new instance (doesn't reuse existing ones on different ports)
    
    Args:
        session_name: Optional name for the session (defaults to session_<port>)
        port: Specific port to use (defaults to finding available port)
    """
    if port is None:
        # Find an available port, starting from the default
        port = find_available_port(_default_port)
        print(f"No port specified, found available port: {port}", file=sys.stderr)

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

    # Force a new instance - always use force_new=True to avoid reusing existing instances
    print(f"Starting new ChimeraX instance on port {port} (force_new=True)", file=sys.stderr)
    success, actual_port = await start_chimerax(port, session_name, force_new=True)
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

@mcp.tool()
async def list_chimerax_commands() -> str:
    """List all available ChimeraX commands.
    
    Use this to discover what commands are available in ChimeraX.
    Once you know the command name, use get_command_documentation() to get detailed syntax.
    """
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
    result += "Use get_command_documentation(command_name) to get detailed documentation for any command."
    return result

@mcp.tool()
async def get_command_documentation(command_name: str) -> str:
    """Get detailed documentation for a specific ChimeraX command.
    
    Use this to learn the correct syntax, arguments, and usage for a specific command.
    
    Args:
        command_name: Name of the ChimeraX command (e.g., 'open', 'color', 'save')
    """
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
