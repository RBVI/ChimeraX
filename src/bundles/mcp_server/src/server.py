import json
import asyncio
import sys
import threading
from typing import Any, Dict, List, Optional
from chimerax.core.logger import Logger


class MCPServer:
    """Model Context Protocol server for ChimeraX integration"""

    def __init__(self, session):
        self.session = session
        self.running = False
        self.server = None
        self.loop = None
        self.thread = None

        # Initialize settings
        from .settings import get_settings
        self.settings = get_settings(session)

    def start(self, port: int = 3001):
        """Start the MCP server on specified port"""
        if self.running:
            return False, "MCP server is already running"

        try:
            # Start the server in a separate thread with its own event loop
            self.thread = threading.Thread(target=self._run_server, args=(port,), daemon=True)
            self.thread.start()
            self.running = True
            return True, f"MCP server started on port {port}"
        except Exception as e:
            return False, f"Failed to start MCP server: {e}"

    def _run_server(self, port):
        """Run the asyncio server in its own thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._start_async_server(port))
        except Exception as e:
            self.session.logger.error(f"MCP server error: {e}")
        finally:
            self.loop.close()

    async def _start_async_server(self, port):
        """Start the async server"""
        self.server = await asyncio.start_server(
            self.handle_client, 'localhost', port
        )

        async with self.server:
            await self.server.serve_forever()

    def stop(self):
        """Stop the MCP server"""
        if not self.running:
            return False, "MCP server is not running"

        self.running = False
        if self.server and self.loop:
            # Schedule server shutdown in the event loop
            asyncio.run_coroutine_threadsafe(self._stop_async_server(), self.loop)

        if self.thread:
            self.thread.join(timeout=2.0)

        self.session.logger.info("MCP server stopped")
        return True, "MCP server stopped"

    async def _stop_async_server(self):
        """Stop the async server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    def status(self):
        """Get server status"""
        if self.running:
            return True, "MCP server is running"
        else:
            return False, "MCP server is stopped"

    async def handle_client(self, reader, writer):
        """Handle incoming MCP client connections"""
        self.reader = reader
        self.writer = writer

        try:
            while self.running:
                data = await reader.readline()
                if not data:
                    break

                try:
                    message = json.loads(data.decode().strip())
                    response = await self.process_message(message)

                    if response:
                        response_data = json.dumps(response) + '\n'
                        writer.write(response_data.encode())
                        await writer.drain()

                except json.JSONDecodeError:
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "Parse error"},
                        "id": None
                    }
                    writer.write((json.dumps(error_response) + '\n').encode())
                    await writer.drain()

        except Exception as e:
            self.session.logger.error(f"MCP client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming MCP messages"""
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")

        try:
            if method == "initialize":
                return await self.handle_initialize(params, msg_id)
            elif method == "tools/list":
                return await self.handle_tools_list(params, msg_id)
            elif method == "tools/call":
                return await self.handle_tools_call(params, msg_id)
            elif method == "resources/list":
                return await self.handle_resources_list(params, msg_id)
            elif method == "resources/read":
                return await self.handle_resources_read(params, msg_id)
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": msg_id
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal error: {e}"},
                "id": msg_id
            }

    async def handle_initialize(self, params: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """Handle MCP initialization"""
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "chimerax-mcp",
                    "version": "0.1.0"
                }
            },
            "id": msg_id
        }

    async def handle_tools_list(self, params: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """List available tools"""
        tools = [
            {
                "name": "run_command",
                "description": "Execute a ChimeraX command",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "ChimeraX command to execute"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "list_models",
                "description": "List all models in the current ChimeraX session",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_model_info",
                "description": "Get detailed information about a specific model",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Model ID to get information for"
                        }
                    },
                    "required": ["model_id"]
                }
            },
            {
                "name": "session_info",
                "description": "Get information about the current ChimeraX session",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

        return {
            "jsonrpc": "2.0",
            "result": {"tools": tools},
            "id": msg_id
        }

    async def handle_tools_call(self, params: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """Handle tool execution"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "run_command":
            return await self.tool_run_command(arguments, msg_id)
        elif tool_name == "list_models":
            return await self.tool_list_models(arguments, msg_id)
        elif tool_name == "get_model_info":
            return await self.tool_get_model_info(arguments, msg_id)
        elif tool_name == "session_info":
            return await self.tool_session_info(arguments, msg_id)
        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                "id": msg_id
            }

    async def tool_run_command(self, arguments: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """Execute a ChimeraX command"""
        command = arguments.get("command")
        if not command:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Missing command parameter"},
                "id": msg_id
            }

        try:
            # Use thread_safe to execute commands on the main thread
            from queue import Queue
            from chimerax.core.logger import StringPlainTextLog

            q = Queue()

            def execute_command():
                try:
                    from chimerax.core.commands import run

                    logger = self.session.logger
                    from chimerax.rest_server.server import ByLevelPlainTextLog
                    log_class = ByLevelPlainTextLog
                    log_class.propagate_to_chimerax = True
                    with ByLevelPlainTextLog(logger) as rest_log:
                        result = run(
                            self.session,
                            command,
                            log = True,
                            #return_json = True,
                            return_list = True
                        )
                        output = rest_log.getvalue()
                        q.put(("success", result, output))
                except Exception as e:
                    q.put(("error", str(e), ""))

            # Execute on main thread
            self.session.ui.thread_safe(execute_command)

            # Get the result
            status, result, output = q.get()

            if status == "success":
                # Format response for AI with both log output and results
                response_text = f"Command executed: {command}\n"
                if output:
                    response_text += f"Log output: {output}\n"
                if result:
                    response_text += f"Return value: {result}\n"

                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": response_text
                            }
                        ]
                    },
                    "id": msg_id
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Command failed: {command}\nError: {result}"
                            }
                        ]
                    },
                    "id": msg_id
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal error: {e}"},
                "id": msg_id
            }

    async def tool_list_models(self, arguments: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """List all models in the session"""
        try:
            from queue import Queue
            q = Queue()

            def get_models():
                try:
                    models = []
                    for model in self.session.models:
                        models.append({
                            "id": model.id_string,
                            "name": model.name,
                            "type": type(model).__name__,
                            "visible": model.display
                        })
                    q.put(("success", models))
                except Exception as e:
                    q.put(("error", str(e)))

            # Execute on main thread
            self.session.ui.thread_safe(get_models)
            status, data = q.get()

            if status == "success":
                models = data
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Models in session ({len(models)} total):\n" +
                                       "\n".join([f"- {m['id']}: {m['name']} ({m['type']}) - {'visible' if m['visible'] else 'hidden'}"
                                                for m in models])
                            }
                        ]
                    },
                    "id": msg_id
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Error listing models: {data}"},
                    "id": msg_id
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Error listing models: {e}"},
                "id": msg_id
            }

    async def tool_get_model_info(self, arguments: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        model_id = arguments.get("model_id")
        if not model_id:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Missing model_id parameter"},
                "id": msg_id
            }

        try:
            model = self.session.models[model_id]
            info = {
                "id": model.id_string,
                "name": model.name,
                "type": type(model).__name__,
                "visible": model.display,
                "position": str(model.scene_position),
            }

            # Add type-specific information
            if hasattr(model, 'num_atoms'):
                info["num_atoms"] = model.num_atoms
            if hasattr(model, 'num_residues'):
                info["num_residues"] = model.num_residues
            if hasattr(model, 'num_chains'):
                info["num_chains"] = model.num_chains

            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Model {model_id} information:\n" +
                                   "\n".join([f"{k}: {v}" for k, v in info.items()])
                        }
                    ]
                },
                "id": msg_id
            }
        except KeyError:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": f"Model {model_id} not found"},
                "id": msg_id
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Error getting model info: {e}"},
                "id": msg_id
            }

    async def tool_session_info(self, arguments: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """Get information about the current session"""
        try:
            info = {
                "models_count": len(self.session.models),
                "logger_level": self.session.logger.level,
                "version": getattr(self.session, 'version', 'unknown'),
            }

            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"ChimeraX Session Information:\n" +
                                   "\n".join([f"{k}: {v}" for k, v in info.items()])
                        }
                    ]
                },
                "id": msg_id
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Error getting session info: {e}"},
                "id": msg_id
            }

    async def handle_resources_list(self, params: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """List available resources"""
        return {
            "jsonrpc": "2.0",
            "result": {"resources": []},
            "id": msg_id
        }

    async def handle_resources_read(self, params: Dict[str, Any], msg_id) -> Dict[str, Any]:
        """Read a specific resource"""
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Resources not implemented"},
            "id": msg_id
        }
