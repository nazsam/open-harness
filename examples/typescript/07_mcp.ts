// Give an agent the tools of any MCP server.
import { Agent, MCPServerStdio, run } from "openharness";

const files = new MCPServerStdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]);
try {
  console.log("tools:", (await files.listTools()).map((t) => t.name));
  const agent = new Agent({ instructions: "You can read files in the current folder.", mcpServers: [files] });
  console.log((await run(agent, "List the files here and summarise the README if there is one.")).output);
} finally {
  await files.close();
}
