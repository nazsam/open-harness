import assert from "node:assert/strict";
import { test } from "node:test";
import { Agent, AnthropicModel, ConfigError, GeminiModel, Message, ModelError, OpenAIModel, getModel, run, runStream, s, tool } from "../dist/index.js";

const getWeather = tool({
  name: "get_weather",
  description: "Get weather.",
  parameters: s.object({ city: s.string().describe("City name.") }),
  execute: ({ city }) => `18C and cloudy in ${city}`,
});

const json = (body, status = 200, headers = {}) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
const sse = (events) =>
  new Response(
    events
      .map((e) => (Array.isArray(e) ? `event: ${e[0]}\ndata: ${JSON.stringify(e[1])}\n\n` : e === "[DONE]" ? "data: [DONE]\n\n" : `data: ${JSON.stringify(e)}\n\n`))
      .join(""),
    { headers: { "content-type": "text/event-stream" } },
  );

function recorder(respond) {
  const calls = [];
  const fetch = async (url, init) => {
    const body = init.body ? JSON.parse(init.body) : undefined;
    calls.push({ url, headers: init.headers, body });
    return respond(calls.length, body, url, init);
  };
  return { calls, fetch };
}

test("openai: tool loop request shape", async () => {
  const rec = recorder((n) =>
    n === 1
      ? json({ choices: [{ finish_reason: "tool_calls", message: { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city": "Paris"}' } }] } }], usage: { prompt_tokens: 50, completion_tokens: 10 } })
      : json({ choices: [{ finish_reason: "stop", message: { role: "assistant", content: "It is 18C in Paris." } }], usage: { prompt_tokens: 70, completion_tokens: 8 } }),
  );
  const model = new OpenAIModel("gpt-test", { apiKey: "sk-test", fetch: rec.fetch });
  const result = await run(new Agent({ model, instructions: "Be brief.", tools: [getWeather] }), "Weather in Paris?");
  assert.equal(result.output, "It is 18C in Paris.");
  assert.equal(result.usage.inputTokens, 120);
  const [first, second] = rec.calls;
  assert.equal(first.url, "https://api.openai.com/v1/chat/completions");
  assert.equal(first.headers.Authorization, "Bearer sk-test");
  assert.deepEqual(first.body.messages[0], { role: "system", content: "Be brief." });
  assert.equal(second.body.messages[2].tool_calls[0].function.arguments, '{"city":"Paris"}');
  assert.deepEqual(second.body.messages[3], { role: "tool", tool_call_id: "call_1", content: "18C and cloudy in Paris" });
});

test("openai: streaming tool call chunks", async () => {
  const rec = recorder((n, body) => {
    assert.equal(body.stream, true);
    return n === 1
      ? sse([
          { choices: [{ delta: { tool_calls: [{ index: 0, id: "c1", function: { name: "get_weather", arguments: "" } }] } }] },
          { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] } }] },
          { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"Rome"}' } }] }, finish_reason: "tool_calls" }] },
          { choices: [], usage: { prompt_tokens: 5, completion_tokens: 5 } },
          "[DONE]",
        ])
      : sse([{ choices: [{ delta: { content: "Rome: " } }] }, { choices: [{ delta: { content: "18C" } }] }, "[DONE]"]);
  });
  const model = new OpenAIModel("gpt-test", { apiKey: "k", fetch: rec.fetch });
  const stream = runStream(new Agent({ model, tools: [getWeather] }), "Rome?");
  const deltas = [];
  for await (const d of stream.text()) deltas.push(d);
  const result = await stream.result();
  assert.deepEqual(deltas, ["Rome: ", "18C"]);
  assert.deepEqual(result.newMessages[1].toolCalls[0].arguments, { city: "Rome" });
});

test("openai: structured output strict detection and compatible presets", async () => {
  const rec = recorder(() => json({ choices: [{ message: { content: '{"answer":"yes"}' } }] }));
  const model = new OpenAIModel("gpt-test", { apiKey: "k", fetch: rec.fetch });
  const out = await run(new Agent({ model, outputType: s.object({ answer: s.string(), note: s.string().optional() }) }), "?");
  assert.deepEqual(out.output, { answer: "yes" });
  assert.equal(rec.calls[0].body.response_format.json_schema.strict, false);
  const ollama = getModel("ollama:llama3.2");
  assert.equal(ollama.baseUrl, "http://localhost:11434/v1");
  assert.equal(ollama.buildBody({ messages: [Message.user("hi")], settings: { maxOutputTokens: 100 } }, false).max_tokens, 100);
  delete process.env.GROQ_API_KEY;
  assert.throws(() => getModel("groq:llama"), ConfigError);
});

test("http: retries 429 then surfaces errors", async () => {
  let n = 0;
  const flaky = async () => (++n < 3 ? json({ error: { message: "slow" } }, 429, { "retry-after": "0" }) : json({ choices: [{ message: { content: "ok" } }] }));
  const model = new OpenAIModel("x", { apiKey: "k", fetch: flaky });
  assert.equal((await run(new Agent({ model }), "hi")).output, "ok");
  assert.equal(n, 3);
  const bad = new OpenAIModel("x", { apiKey: "k", maxRetries: 0, fetch: async () => json({ error: { message: "bad model" } }, 400) });
  await assert.rejects(run(new Agent({ model: bad }), "hi"), (e) => e instanceof ModelError && /bad model/.test(e.message));
});

test("anthropic: tool loop preserves thinking blocks", async () => {
  const rec = recorder((n) =>
    n === 1
      ? json({ stop_reason: "tool_use", usage: { input_tokens: 30, output_tokens: 9 }, content: [
          { type: "thinking", thinking: "need weather", signature: "sig123" },
          { type: "text", text: "Checking." },
          { type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "Oslo" } },
        ] })
      : json({ stop_reason: "end_turn", usage: { input_tokens: 60, output_tokens: 5 }, content: [{ type: "text", text: "Oslo is 18C." }] }),
  );
  const model = new AnthropicModel("claude-test", { apiKey: "ak", fetch: rec.fetch });
  const result = await run(new Agent({ model, instructions: "Sys.", tools: [getWeather] }), "Oslo?");
  assert.equal(result.output, "Oslo is 18C.");
  const [first, second] = rec.calls;
  assert.equal(first.headers["x-api-key"], "ak");
  assert.equal(first.headers["anthropic-version"], "2023-06-01");
  assert.equal(first.body.system, "Sys.");
  assert.equal(first.body.max_tokens, 16000);
  assert.deepEqual(second.body.messages[1].content[0], { type: "thinking", thinking: "need weather", signature: "sig123" });
  assert.deepEqual(second.body.messages[2].content[0], { type: "tool_result", tool_use_id: "toolu_1", content: "18C and cloudy in Oslo" });
});

test("anthropic: streaming", async () => {
  const model = new AnthropicModel("claude-test", {
    apiKey: "ak",
    fetch: async () =>
      sse([
        ["message_start", { type: "message_start", message: { usage: { input_tokens: 12, output_tokens: 1 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hel" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "lo" } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["content_block_start", { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "t1", name: "get_weather", input: {} } }],
        ["content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '{"city": "L' } }],
        ["content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: 'ima"}' } }],
        ["content_block_stop", { type: "content_block_stop", index: 1 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 20 } }],
      ]),
  });
  const events = [];
  for await (const e of model.stream({ messages: [Message.user("hi")] })) events.push(e);
  assert.deepEqual(events.filter((e) => e.type === "text_delta").map((e) => e.delta), ["Hel", "lo"]);
  const done = events.at(-1).response;
  assert.equal(done.message.content, "Hello");
  assert.deepEqual(done.message.toolCalls[0].arguments, { city: "Lima" });
  assert.equal(done.usage.outputTokens, 20);
});

test("anthropic: merges tool results and sends output_config", () => {
  const model = new AnthropicModel("c", { apiKey: "k" });
  const msgs = [
    Message.user("q"),
    Message.assistant("", [{ id: "a", name: "x", arguments: {} }, { id: "b", name: "y", arguments: {} }]),
    Message.tool("a", "x", "1"),
    Message.tool("b", "y", "2", true),
    Message.user("more"),
  ];
  const body = model.buildBody({ system: "s", messages: msgs, outputSchema: { name: "o", schema: { type: "object" } } }, false);
  assert.deepEqual(body.messages.map((m) => m.role), ["user", "assistant", "user"]);
  assert.deepEqual(body.messages[2].content.map((b) => b.type), ["tool_result", "tool_result", "text"]);
  assert.deepEqual(body.output_config, { format: { type: "json_schema", schema: { type: "object" } } });
});

test("gemini: tool loop keeps thought signature", async () => {
  const rec = recorder((n) =>
    n === 1
      ? json({ candidates: [{ content: { role: "model", parts: [{ functionCall: { name: "get_weather", args: { city: "Lagos" } }, thoughtSignature: "abc" }] } }], usageMetadata: { promptTokenCount: 20, candidatesTokenCount: 4, thoughtsTokenCount: 6 } })
      : json({ candidates: [{ content: { parts: [{ text: "Lagos: 18C" }] } }] }),
  );
  const model = new GeminiModel("gemini-test", { apiKey: "gk", fetch: rec.fetch });
  const result = await run(new Agent({ model, instructions: "Sys.", tools: [getWeather] }), "Lagos?");
  assert.equal(result.output, "Lagos: 18C");
  const [first, second] = rec.calls;
  assert.match(first.url, /\/models\/gemini-test:generateContent$/);
  assert.equal(first.headers["x-goog-api-key"], "gk");
  assert.deepEqual(first.body.systemInstruction, { parts: [{ text: "Sys." }] });
  assert.deepEqual(first.body.tools[0].functionDeclarations[0].parametersJsonSchema.required, ["city"]);
  assert.deepEqual(second.body.contents[1], { role: "model", parts: [{ functionCall: { name: "get_weather", args: { city: "Lagos" } }, thoughtSignature: "abc" }] });
  assert.deepEqual(second.body.contents[2].parts[0].functionResponse, { name: "get_weather", response: { result: "18C and cloudy in Lagos" } });
  assert.equal(result.usage.outputTokens, 10);
});

test("gemini: streaming", async () => {
  const model = new GeminiModel("gemini-test", {
    apiKey: "gk",
    fetch: async (url) => {
      assert.match(url, /alt=sse/);
      return sse([{ candidates: [{ content: { parts: [{ text: "Hi " }] } }] }, { candidates: [{ content: { parts: [{ text: "there" }] }, finishReason: "STOP" }] }]);
    },
  });
  const stream = runStream(new Agent({ model }), "hi");
  const out = [];
  for await (const d of stream.text()) out.push(d);
  assert.deepEqual(out, ["Hi ", "there"]);
  assert.equal((await stream.result()).output, "Hi there");
});

test("history moves between providers", () => {
  const assistant = Message.assistant("Checking.", [{ id: "toolu_1", name: "get_weather", arguments: { city: "Oslo" } }]);
  assistant.raw = { provider: "anthropic", data: [{ type: "thinking", thinking: "x", signature: "s" }] };
  const req = { system: "s", messages: [Message.user("Oslo?"), assistant, Message.tool("toolu_1", "get_weather", "18C")] };
  const oa = new OpenAIModel("gpt", { apiKey: "k" }).buildBody(req, false);
  assert.equal(oa.messages[2].tool_calls[0].id, "toolu_1");
  const gm = new GeminiModel("g", { apiKey: "k" }).buildBody(req);
  assert.deepEqual(gm.contents[1].parts[1].functionCall, { name: "get_weather", args: { city: "Oslo" }, id: "toolu_1" });
  assert.ok(!JSON.stringify(gm).includes("thinking"));
});

test("getModel inference", () => {
  process.env.ANTHROPIC_API_KEY = "a";
  delete process.env.OPENHARNESS_MODEL;
  assert.equal(getModel("claude-sonnet-5").provider, "anthropic");
  assert.equal(getModel().model, "claude-sonnet-5");
  assert.throws(() => getModel("mystery-model"), ConfigError);
  delete process.env.ANTHROPIC_API_KEY;
});
