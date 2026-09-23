---
title: MCP servers
parent: Guides
nav_order: 9
---

# MCP servers

Connect any [Model Context Protocol](https://modelcontextprotocol.io) server and its tools become agent tools.

```python
from openharness import Agent, MCPServerHTTP, MCPServerStdio

files = MCPServerStdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"])
github = MCPServerHTTP("https://mcp.example.com/mcp", headers={"Authorization": f"Bearer {token}"})

async with files:
    agent = Agent(instructions="You can read and write files in ./workspace.", mcp_servers=[files, github])
    print((await run(agent, "Create notes.md with a summary of README.md")).output)
```

```ts
import { MCPServerHTTP, MCPServerStdio } from "openharness";

const files = new MCPServerStdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]);
const github = new MCPServerHTTP("https://mcp.example.com/mcp", { headers: { Authorization: `Bearer ${token}` } });
const agent = new Agent({ mcpServers: [files, github] });
try {
  await run(agent, "Create notes.md with a summary of README.md");
} finally {
  await files.close();
}
```

From the terminal: `openharness chat --mcp "npx -y @modelcontextprotocol/server-filesystem ." --mcp-http https://...`

## Options

| Option | Python | TypeScript |
|---|---|---|
| Only some tools | `tool_filter=lambda name: name.startswith("read")` | `toolFilter: (n) => n.startsWith("read")` |
| Avoid name clashes | `tool_prefix="fs_"` | `toolPrefix: "fs_"` |
| Require approval for every tool | `needs_approval=True` | `needsApproval: true` |
| Per-call timeout | `timeout=60` | `timeoutSeconds: 60` |
| Re-list tools each time | `cache_tools=False` | `cacheTools: false` |
| Environment / working directory (stdio) | `env={...}, cwd="..."` | `{ env, cwd }` |

## Protocol versions

The client speaks MCP revision **2026-07-28**, which is stateless: each request carries its protocol version and client
info in `_meta`, and HTTP requests carry `MCP-Protocol-Version`, `Mcp-Method` and `Mcp-Name` headers (plus any
`x-mcp-header` parameter headers a tool defines).

It also works with servers on older, handshake-based revisions (2025-11-25 and earlier), following the spec's dual-era
rules:

* **stdio**: the client probes with `server/discover`. A modern reply (or modern error) keeps it modern; any other
  error or a timeout falls back to `initialize`.
* **HTTP**: the client tries a modern request. A `4xx` without a recognised modern error body falls back to
  `initialize` and uses `Mcp-Session-Id` from then on.

`server.era` tells you which one was used.

Tool results are converted to text for the model: text content is joined, images and resources become short
placeholders, and `structuredContent` is used when there is no text. `isError` results are sent back as tool errors.

{: .warning }
MCP tools run with whatever access the server has. Only connect servers you trust, and consider `needs_approval=True`
for servers that can write or delete.
