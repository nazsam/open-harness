// Tiny MCP stdio server used by the tests. `--mode modern` or `--mode legacy`.
import { createInterface } from "node:readline";

const mode = process.argv.includes("--mode") ? process.argv[process.argv.indexOf("--mode") + 1] : "modern";
const TOOLS = [
  {
    name: "add",
    description: "Add two integers",
    inputSchema: { type: "object", properties: { a: { type: "integer" }, b: { type: "integer" } }, required: ["a", "b"] },
  },
  { name: "fail", description: "Always fails", inputSchema: { type: "object", properties: {} } },
];
let initialized = false;
const reply = (id, result, error) => process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id, ...(error ? { error } : { result }) }) + "\n");

createInterface({ input: process.stdin }).on("line", (line) => {
  const msg = JSON.parse(line);
  const { method, id } = msg;
  const params = msg.params ?? {};
  if (id === undefined) return;
  if (mode === "modern") {
    if (method === "server/discover")
      return reply(id, { resultType: "complete", supportedVersions: ["2026-07-28"], serverInfo: { name: "test", version: "1" } });
    if (params._meta?.["io.modelcontextprotocol/protocolVersion"] !== "2026-07-28") return reply(id, null, { code: -32602, message: "missing _meta" });
  } else {
    if (method === "initialize") {
      initialized = true;
      return reply(id, { protocolVersion: "2025-11-25", capabilities: { tools: {} }, serverInfo: { name: "legacy-test", version: "1" } });
    }
    if (!initialized) return reply(id, null, { code: -32601, message: `Method not found: ${method}` });
  }
  if (method === "tools/list") return reply(id, { tools: TOOLS });
  if (method === "tools/call") {
    const { name, arguments: args = {} } = params;
    if (name === "add") return reply(id, { content: [{ type: "text", text: String(args.a + args.b) }], isError: false });
    return reply(id, { content: [{ type: "text", text: "boom" }], isError: true });
  }
  reply(id, null, { code: -32601, message: "Method not found" });
});
