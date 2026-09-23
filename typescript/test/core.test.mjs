import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { setTimeout as setTimeoutP } from "node:timers/promises";
import {
  Agent, CancelToken, FakeModel, FileMemory, FileSession, GuardrailTripped, InMemoryMemory, InMemorySession, MaxToolCallsExceeded,
  MaxTurnsExceeded, MemoryProcessor, OutputValidationError, RunCancelled, RunTimeout, TokenBudgetExceeded, ToolError, call,
  inputGuardrail, maxLength, outputGuardrail, pii, run, runStream, s, tool,
} from "../dist/index.js";

const add = tool({
  name: "add",
  description: "Add two integers.",
  parameters: s.object({ a: s.integer().describe("First number."), b: s.integer() }),
  execute: ({ a, b }) => a + b,
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const tmp = () => mkdtempSync(join(tmpdir(), "oh-"));

test("schema builder produces JSON Schema", () => {
  const Weather = s.object({ city: s.string(), temp: s.number(), note: s.string().optional(), kind: s.enum(["a", "b"]) });
  assert.deepEqual(Weather.jsonSchema.required, ["city", "temp", "kind"]);
  assert.equal(Weather.jsonSchema.additionalProperties, false);
  assert.deepEqual(Weather.jsonSchema.properties.kind, { enum: ["a", "b"], type: "string" });
  assert.deepEqual(add.parameters.properties.a, { type: "integer", description: "First number." });
});

test("basic tool loop and usage", async () => {
  const model = new FakeModel([call("add", { a: 2, b: 40 }), "The answer is 42."]);
  const result = await run(new Agent({ model, tools: [add] }), "What is 2 + 40?");
  assert.equal(result.output, "The answer is 42.");
  assert.equal(result.turns, 2);
  assert.deepEqual(result.newMessages.map((m) => m.role), ["user", "assistant", "tool", "assistant"]);
  assert.equal(result.newMessages[2].content, "42");
  assert.equal(result.usage.requests, 2);
});

test("parallel tools run concurrently", async () => {
  const slow = tool({ name: "slow", description: "Slow.", parameters: s.object({ n: s.integer() }), execute: async ({ n }) => (await sleep(200), n) });
  const model = new FakeModel([{ toolCalls: [0, 1, 2, 3, 4].map((n) => ({ name: "slow", arguments: { n } })) }, "done"]);
  const t0 = Date.now();
  const result = await run(new Agent({ model, tools: [slow] }), "go");
  assert.ok(Date.now() - t0 < 800);
  assert.deepEqual(result.newMessages.filter((m) => m.role === "tool").map((m) => m.content), ["0", "1", "2", "3", "4"]);
});

test("tool errors go back to the model", async () => {
  const broken = tool({ name: "broken", description: "Broken.", execute: () => { throw new ToolError("x is not allowed"); } });
  const model = new FakeModel([call("broken"), call("nope"), call("add", { a: "oops", b: 1 }), "ok"]);
  const result = await run(new Agent({ model, tools: [broken, add] }), "go");
  const errors = result.newMessages.filter((m) => m.role === "tool");
  assert.ok(errors.every((m) => m.isError));
  assert.match(errors[0].content, /x is not allowed/);
  assert.match(errors[1].content, /unknown tool 'nope'/);
  assert.match(errors[2].content, /Invalid arguments/);
});

test("deps reach tools and dynamic instructions", async () => {
  const whoami = tool({ name: "whoami", description: "User.", execute: (_a, ctx) => ctx.deps.user });
  const model = new FakeModel([call("whoami"), "hi"]);
  const agent = new Agent({ model, tools: [whoami], instructions: (ctx) => `User is ${ctx.deps.user}.` });
  const result = await run(agent, "who", { deps: { user: "sam" } });
  assert.equal(result.newMessages[2].content, "sam");
  assert.equal(model.requests[0].system, "User is sam.");
});

test("structured output with retry", async () => {
  const Weather = s.object({ city: s.string(), tempC: s.number(), conditions: s.enum(["sunny", "cloudy", "rain"]) });
  const model = new FakeModel(["not json", { json: { city: "Oslo", tempC: 3, conditions: "snow" } }, '```json\n{"city":"Oslo","tempC":3,"conditions":"rain"}\n```']);
  const result = await run(new Agent({ model, outputType: Weather }), "weather?");
  assert.deepEqual(result.output, { city: "Oslo", tempC: 3, conditions: "rain" });
  assert.equal(result.turns, 3);
  assert.deepEqual(model.requests[0].outputSchema.schema.required, ["city", "tempC", "conditions"]);
});

test("structured output non-object is wrapped and unwrapped", async () => {
  const model = new FakeModel([{ json: { value: ["a", "b"] } }]);
  const result = await run(new Agent({ model, outputType: s.array(s.string()) }), "list");
  assert.deepEqual(result.output, ["a", "b"]);
});

test("structured output gives up", async () => {
  await assert.rejects(run(new Agent({ model: new FakeModel([], { defaultReply: "nope" }), outputType: s.object({ a: s.string() }) }), "?", { limits: { maxOutputRetries: 1 } }), OutputValidationError);
});

test("streaming events", async () => {
  const model = new FakeModel([call("add", { a: 1, b: 1 }), "Two it is."], { chunkSize: 3 });
  const stream = runStream(new Agent({ model, tools: [add] }), "1+1");
  const types = [];
  let text = "";
  for await (const ev of stream) {
    types.push(ev.type);
    if (ev.type === "text_delta") text += ev.data.delta;
  }
  const result = await stream.result();
  assert.equal(text, "Two it is.");
  assert.equal(result.output, "Two it is.");
  assert.equal(types[0], "agent_start");
  assert.equal(types.at(-1), "run_end");
  assert.ok(types.indexOf("tool_call") < types.indexOf("tool_result"));
});

for (const kind of ["memory", "file"]) {
  test(`${kind} session remembers history`, async () => {
    const session = kind === "memory" ? new InMemorySession() : new FileSession(join(tmp(), "s.jsonl"));
    const model = new FakeModel(["Nice to meet you", (req) => `seen ${req.messages.length} messages`]);
    const agent = new Agent({ model });
    await run(agent, "I am Alice", { session });
    const result = await run(agent, "who am I?", { session });
    assert.equal(result.output, "seen 3 messages");
    assert.equal((await session.getMessages()).length, 4);
    await session.clear();
    assert.deepEqual(await session.getMessages(), []);
  });
}

test("memory tools and injection", async () => {
  const path = join(tmp(), "mem.json");
  const model = new FakeModel([call("remember", { fact: "The user prefers metric units" }), "Saved.", "ok"]);
  const agent = new Agent({ model, memory: new FileMemory(path) });
  await run(agent, "remember I like metric");
  assert.deepEqual((await new FileMemory(path).list()).map((m) => m.text), ["The user prefers metric units"]);
  await run(agent, "what units do I prefer?");
  assert.match(model.requests.at(-1).system, /prefers metric units/);
  assert.deepEqual(model.requests.at(-1).tools.map((t) => t.name), ["remember", "recall"]);
  const mem = new InMemoryMemory();
  await mem.add("Sam lives in Toronto");
  await mem.add("Favourite language is Python");
  assert.equal((await mem.search("which city does Sam live in"))[0].text, "Sam lives in Toronto");
});

test("guardrails: block, redact, output", async () => {
  const model = new FakeModel(["Got it", "the SECRET is 42"]);
  await assert.rejects(run(new Agent({ model: new FakeModel(), inputGuardrails: [maxLength(5)] }), "this is too long"), GuardrailTripped);
  const noSecrets = outputGuardrail("no_secrets", (_ctx, out) => out.includes("SECRET"));
  const agent = new Agent({ model, inputGuardrails: [pii()], outputGuardrails: [noSecrets] });
  await run(agent, "mail me at sam@example.com or 416-555-0199");
  assert.equal(model.requests[0].messages[0].content, "mail me at [EMAIL] or [PHONE]");
  await assert.rejects(run(agent, "tell me"), GuardrailTripped);
  const homework = inputGuardrail("homework", async (_ctx, text) => text.includes("homework"));
  await assert.rejects(run(new Agent({ model: new FakeModel(), inputGuardrails: [homework] }), "do my homework"), GuardrailTripped);
});

test("limits: turns, tool calls, tokens", async () => {
  const loop = () => new FakeModel(Array(10).fill(call("add", { a: 1, b: 1 })));
  await assert.rejects(run(new Agent({ model: loop(), tools: [add] }), "x", { limits: { maxTurns: 3 } }), (e) => e instanceof MaxTurnsExceeded && e.partial.turns === 3);
  await assert.rejects(run(new Agent({ model: loop(), tools: [add] }), "x", { limits: { maxToolCalls: 2 } }), MaxToolCallsExceeded);
  await assert.rejects(run(new Agent({ model: loop(), tools: [add] }), "x", { limits: { maxTotalTokens: 25 } }), TokenBudgetExceeded);
});

test("timeout and cancellation interrupt slow tools", async () => {
  // Tools receive ctx.signal, which aborts when the run is cancelled or times out.
  const sleepy = tool({ name: "sleepy", description: "Sleep.", execute: async (_a, ctx) => (await setTimeoutP(10000, null, { signal: ctx.signal }), "late") });
  await assert.rejects(run(new Agent({ model: new FakeModel([call("sleepy")]), tools: [sleepy] }), "go", { limits: { timeoutSeconds: 0.3 } }), RunTimeout);
  const token = new CancelToken();
  setTimeout(() => token.cancel("user pressed stop"), 200);
  await assert.rejects(run(new Agent({ model: new FakeModel([call("sleepy")]), tools: [sleepy] }), "go", { cancelToken: token }), (e) => e instanceof RunCancelled && /user pressed stop/.test(e.message));
  const ac = new AbortController();
  setTimeout(() => ac.abort("bye"), 100);
  await assert.rejects(run(new Agent({ model: new FakeModel([call("sleepy")]), tools: [sleepy] }), "go", { signal: ac.signal }), RunCancelled);
});

test("tool approval", async () => {
  const del = tool({ name: "delete_file", description: "Delete.", parameters: s.object({ path: s.string() }), needsApproval: true, execute: ({ path }) => `deleted ${path}` });
  const model = new FakeModel([call("delete_file", { path: "tmp.txt" }), call("delete_file", { path: "/etc/passwd" }), "done"]);
  const result = await run(new Agent({ model, tools: [del] }), "clean", { approve: (_ctx, tc) => tc.arguments.path === "tmp.txt" });
  const outs = result.newMessages.filter((m) => m.role === "tool");
  assert.equal(outs[0].content, "deleted tmp.txt");
  assert.ok(outs[1].isError && /did not approve/.test(outs[1].content));
});

test("handoffs and agents as tools", async () => {
  const billingModel = new FakeModel(["Your refund is on its way."]);
  const billing = new Agent({ name: "Billing", description: "Handles refunds.", model: billingModel });
  const triageModel = new FakeModel([call("transfer_to_billing")]);
  const result = await run(new Agent({ name: "Triage", model: triageModel, handoffs: [billing] }), "I want a refund");
  assert.equal(result.output, "Your refund is on its way.");
  assert.equal(result.lastAgent, billing);
  assert.deepEqual(triageModel.requests[0].tools.map((t) => t.name), ["transfer_to_billing"]);

  const researcher = new Agent({ name: "Researcher", model: new FakeModel(["Paris is the capital of France."]) });
  const lead = new Agent({ name: "Lead", model: new FakeModel([call("researcher", { input: "capital?" }), "It is Paris."]), tools: [researcher.asTool()] });
  const r2 = await run(lead, "Capital of France?");
  assert.equal(r2.output, "It is Paris.");
  assert.equal(r2.newMessages[2].content, "Paris is the capital of France.");
  assert.equal(r2.usage.requests, 3);
});

test("tracing spans", async () => {
  const proc = new MemoryProcessor();
  await run(new Agent({ model: new FakeModel([call("add", { a: 1, b: 2 }), "3"]), tools: [add] }), "1+2", { trace: proc });
  const spans = proc.traces[0].spans;
  const kinds = spans.map((sp) => sp.kind);
  assert.equal(kinds[0], "run");
  assert.equal(kinds.filter((k) => k === "model").length, 2);
  const toolSpan = spans.find((sp) => sp.kind === "tool");
  assert.equal(toolSpan.parentId, spans.find((sp) => sp.kind === "model").id);
  assert.equal(toolSpan.attributes.output, "3");
  assert.ok(spans.every((sp) => sp.end !== undefined));
});
