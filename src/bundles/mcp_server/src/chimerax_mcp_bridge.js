#!/usr/bin/env node

/**
 * ChimeraX MCP Client Bridge
 * 
 * This bridges Claude (via MCP SDK) to the ChimeraX MCP server running inside ChimeraX.
 * 
 * Prerequisites:
 * 1. Install the ChimeraX MCP server bundle in ChimeraX
 * 2. Start ChimeraX
 * 3. In ChimeraX, run: mcp start 3001
 * 4. Configure this bridge in your Claude Desktop config
 * 
 * Usage with Claude Desktop:
 * Add to your claude_desktop_config.json:
 * {
 *   "mcpServers": {
 *     "chimerax": {
 *       "command": "node",
 *       "args": ["/path/to/chimerax-mcp-bridge.js"],
 *       "env": {
 *         "CHIMERAX_MCP_HOST": "localhost",
 *         "CHIMERAX_MCP_PORT": "3001"
 *       }
 *     }
 *   }
 * }
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import net from 'net';
import { createInterface } from 'readline';

// Get ChimeraX MCP server connection details from environment
const CHIMERAX_HOST = process.env.CHIMERAX_MCP_HOST || 'localhost';
const CHIMERAX_PORT = parseInt(process.env.CHIMERAX_MCP_PORT || '3001');

class ChimeraXClient {
  constructor() {
    this.socket = null;
    this.messageId = 0;
    this.pendingRequests = new Map();
    this.readline = null;
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.socket = net.createConnection({ host: CHIMERAX_HOST, port: CHIMERAX_PORT }, () => {
        console.error(`Connected to ChimeraX MCP server at ${CHIMERAX_HOST}:${CHIMERAX_PORT}`);

        // Set up readline interface for line-based JSON-RPC
        this.readline = createInterface({
          input: this.socket,
          crlfDelay: Infinity
        });

        this.readline.on('line', (line) => {
          try {
            const response = JSON.parse(line);
            this.handleResponse(response);
          } catch (error) {
            console.error('Failed to parse response:', error);
          }
        });

        resolve();
      });

      this.socket.on('error', (error) => {
        console.error('Socket error:', error);
        reject(error);
      });

      this.socket.on('close', () => {
        console.error('Connection to ChimeraX closed');
      });
    });
  }

  handleResponse(response) {
    const id = response.id;
    if (id !== null && id !== undefined && this.pendingRequests.has(id)) {
      const { resolve, reject } = this.pendingRequests.get(id);
      this.pendingRequests.delete(id);

      if (response.error) {
        reject(new Error(response.error.message));
      } else {
        resolve(response.result);
      }
    }
  }

  async sendRequest(method, params = {}) {
    if (!this.socket || this.socket.destroyed) {
      await this.connect();
    }

    const id = ++this.messageId;
    const message = {
      jsonrpc: '2.0',
      method,
      params,
      id
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });

      // Set a timeout for the request
      const timeout = setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error('Request timeout'));
        }
      }, 30000); // 30 second timeout

      // Clear timeout on resolution
      const originalResolve = resolve;
      const originalReject = reject;
      this.pendingRequests.set(id, {
        resolve: (result) => {
          clearTimeout(timeout);
          originalResolve(result);
        },
        reject: (error) => {
          clearTimeout(timeout);
          originalReject(error);
        }
      });

      this.socket.write(JSON.stringify(message) + '\n');
    });
  }

  async initialize() {
    return this.sendRequest('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: {
        name: 'claude-chimerax-bridge',
        version: '1.0.0'
      }
    });
  }

  async listTools() {
    return this.sendRequest('tools/list');
  }

  async callTool(name, args) {
    return this.sendRequest('tools/call', {
      name,
      arguments: args
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.end();
    }
  }
}

// Global ChimeraX client instance
let chimeraXClient = null;

/**
 * Create and configure the MCP server (bridge to Claude)
 */
const server = new Server(
  {
    name: 'chimerax-bridge',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

/**
 * Handler for listing available tools
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  try {
    // Connect to ChimeraX if not already connected
    if (!chimeraXClient) {
      chimeraXClient = new ChimeraXClient();
      await chimeraXClient.connect();
      await chimeraXClient.initialize();
    }

    // Get tools from ChimeraX MCP server
    const result = await chimeraXClient.listTools();
    return { tools: result.tools };
  } catch (error) {
    console.error('Error listing tools:', error);
    // Return default tools if connection fails
    return {
      tools: [
        {
          name: 'run_command',
          description: 'Execute a ChimeraX command. Connection error - ChimeraX may not be running or MCP server not started.',
          inputSchema: {
            type: 'object',
            properties: {
              command: {
                type: 'string',
                description: 'ChimeraX command to execute',
              },
            },
            required: ['command'],
          },
        },
      ],
    };
  }
});

/**
 * Handler for tool execution
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    // Connect to ChimeraX if not already connected
    if (!chimeraXClient) {
      chimeraXClient = new ChimeraXClient();
      await chimeraXClient.connect();
      await chimeraXClient.initialize();
    }

    const toolName = request.params.name;
    const args = request.params.arguments || {};

    // Forward the tool call to ChimeraX
    const result = await chimeraXClient.callTool(toolName, args);

    // Return the result from ChimeraX
    return {
      content: result.content || [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    console.error('Error calling tool:', error);
    return {
      content: [
        {
          type: 'text',
          text: `Error: ${error.message}. Make sure ChimeraX is running and the MCP server is started (run 'mcp start 3001' in ChimeraX).`,
        },
      ],
      isError: true,
    };
  }
});

/**
 * Start the server
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('ChimeraX MCP bridge running on stdio');
  console.error(`Will connect to ChimeraX at ${CHIMERAX_HOST}:${CHIMERAX_PORT}`);
}

// Cleanup on exit
process.on('SIGINT', () => {
  if (chimeraXClient) {
    chimeraXClient.disconnect();
  }
  process.exit(0);
});

process.on('SIGTERM', () => {
  if (chimeraXClient) {
    chimeraXClient.disconnect();
  }
  process.exit(0);
});

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
