import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { PassThrough } from "node:stream";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Agent, FakeModel, Harness, MCPServerHTTP, MCPServerStdio, call, run, safeEval } from "../dist/index.js";

const SERVER = fileURLToPath(new URL("./mcp-test-server.mjs", import.meta.url));
const CLI = fileURLToPath(new URL("../dist/cli.js", import.meta.url));

for (const mode of ["modern", "legacy"]) {
  test(`mcp stdio (${mode})`, async () => {
    const server = new MCPServerStdio(process.execPath, [SERVER, "--mode", mode], { probeTimeoutSeconds: 2 });
    try {
      const tools = await server.listTools();
      assert.equal(server.era, mode);
      assert.deepEqual(tools.map((t) => t.name), ["add", "fail"]);
      const model = new FakeModel([call("add", { a: 20, b: 22 }), call("fail"), "done"]);
      const result = await run(new Agent({ model, mcpServers: [server] }), "add");
      const outs = result.newMessages.filter((m) => m.role === "tool");
      assert.equal(outs[0].content, "42");
      assert.ok(outs[1].isError && /boom/.test(outs[1].content));
    } finally {
      await server.close();
    }
  });
}

function httpServer(mode) {
  const state = { requests: [] };
  const fetch = async (_url, init) => {
    const h = Object.fromEntries(Object.entries(init.headers ?? {}).map(([k, v]) => [k.toLowerCase(), v]));
    const body = init.body ? JSON.parse(init.body) : {};
    state.requests.push({ h, body });
    const { method, params = {}, id } = body;
    const reply = (o, status = 200, headers = {}) => new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json", ...headers } });
    if (init.method === "DELETE") return new Response(null, { status: 200 });
    if (mode === "modern") {
      if (h["mcp-method"] !== method) return reply({ jsonrpc: "2.0", id, error: { code: -32020, message: "Header mismatch" } }, 400);
      if (method === "server/discover") return reply({ jsonrpc: "2.0", id, result: { resultType: "complete", supportedVersions: ["2026-07-28"] } });
      assert.equal(params._meta["io.modelcontextprotocol/protocolVersion"], "2026-07-28");
    } else {
      if (method === "initialize") return reply({ jsonrpc: "2.0", id, result: { protocolVersion: "2025-11-25", capabilities: {} } }, 200, { "mcp-session-id": "sess-1" });
      if (id === undefined) return new Response(null, { status: 202 });
      if (h["mcp-session-id"] !== "sess-1") return new Response("Bad Request: no session", { status: 400 });
    }
    if (method === "tools/list")
      return reply({ jsonrpc: "2.0", id, result: { tools: [
        { name: "echo", inputSchema: { type: "object", properties: { text: { type: "string" }, region: { type: "string", "x-mcp-header": "Region" } } } },
        { name: "bad", inputSchema: { type: "object", properties: { n: { type: "number", "x-mcp-header": "N" } } } },
      ] } });
    if (method === "tools/call") {
      if (mode === "modern") assert.equal(h["mcp-name"], params.name);
      const text = `${params.arguments.text}|${h["mcp-param-region"] ?? "-"}`;
      const body2 = `event: message\ndata: ${JSON.stringify({ jsonrpc: "2.0", method: "notifications/progress" })}\n\ndata: ${JSON.stringify({ jsonrpc: "2.0", id, result: { content: [{ type: "text", text }] } })}\n\n`;
      return new Response(body2, { headers: { "content-type": "text/event-stream" } });
    }
    return reply({ jsonrpc: "2.0", id, error: { code: -32601, message: "not found" } }, 404);
  };
  return { fetch, state };
}

for (const mode of ["modern", "legacy"]) {
  test(`mcp http (${mode})`, async () => {
    const { fetch } = httpServer(mode);
    const server = new MCPServerHTTP("https://mcp.example.com/mcp", { fetch });
    const tools = await server.listTools();
    assert.equal(server.era, mode);
    assert.deepEqual(tools.map((t) => t.name), ["echo"]);
    const res = await server.callTool("echo", { text: "hi", region: "Café" });
    assert.equal(res.content[0].text, `hi|${mode === "modern" ? "=?base64?Q2Fmw6k=?=" : "-"}`);
    await server.close();
  });
}

test("harness keeps conversation and streams", async () => {
  const h = new Harness({ model: new FakeModel(["Hello Sam!", (r) => `${r.messages.length} messages so far`]) });
  assert.equal(await h.ask("I'm Sam"), "Hello Sam!");
  let text = "";
  for await (const c of h.stream("again")) text += c;
  assert.equal(text, "3 messages so far");
  assert.equal(h.usage.requests, 2);
  assert.deepEqual(h.agent.tools.map((t) => t.name), ["calculator", "current_time"]);
});

test("terminal chat session", async () => {
  const h = new Harness({ model: new FakeModel([call("calculator", { expression: "6*7" }), "It's 42."]) });
  const input = new PassThrough();
  const output = new PassThrough();
  let text = "";
  output.on("data", (d) => (text += d));
  const done = h.chat({ input, output });
  input.write("what is 6*7?\n");
  await new Promise((r) => setTimeout(r, 100));
  input.end("/usage\n/exit\n");
  await done;
  assert.match(text, /-> calculator/);
  assert.match(text, /<- ok: 42/);
  assert.match(text, /It's 42\./);
  assert.match(text, /"requests":2/);
});

test("cli run and models", () => {
  assert.equal(execFileSync(process.execPath, [CLI, "run", "--model", "fake:demo", "hello", "there"], { input: "" }).toString().trim(), "echo: hello there");
  assert.match(execFileSync(process.execPath, [CLI, "models"]).toString(), /ollama/);
});

test("calculator is safe", () => {
  assert.equal(safeEval("2 ** 10 + sqrt(16)"), 1028);
  assert.equal(safeEval("-(3 + 4) * 2 // 3"), -5);
  assert.throws(() => safeEval("process.exit(1)"));
});
