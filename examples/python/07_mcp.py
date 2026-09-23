"""Give an agent the tools of any MCP server. Requires Node.js for this server."""

import asyncio

from openharness import Agent, MCPServerStdio, run


async def main() -> None:
    files = MCPServerStdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
    async with files:
        print("tools:", [t.name for t in await files.list_tools()])
        agent = Agent(instructions="You can read files in the current folder.", mcp_servers=[files])
        print((await run(agent, "List the files here and summarise the README if there is one.")).output)


asyncio.run(main())
