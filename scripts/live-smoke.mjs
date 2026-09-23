// Live smoke test against a real provider. Run by the "Live test" workflow.
//   OPENHARNESS_MODEL=anthropic:claude-sonnet-5 ANTHROPIC_API_KEY=... node scripts/live-smoke.mjs
import assert from "node:assert/strict";
import { Agent, InMemorySession, calculator, run, runStream, s } from "../typescript/dist/index.js";

const model = process.env.OPENHARNESS_MODEL ?? "anthropic:claude-sonnet-5";

const agent = new Agent({
  instructions: "Use the calculator for all arithmetic.",
  model,
  tools: [calculator],
  outputType: s.object({ result: s.integer(), explanation: s.string() }),
});
const r1 = await run(agent, "What is 1234 * 5678?");
assert.equal(r1.output.result, 7006652);
assert.ok(r1.newMessages.some((m) => m.role === "tool"), "calculator was not called");
console.log(`[ok] tools + structured output: ${r1.output.result} (${r1.turns} turns, ${r1.usage.totalTokens} tokens)`);

const chat = new Agent({ instructions: "Be brief.", model, tools: [calculator] });
let text = "";
for await (const d of runStream(chat, "Use the calculator to compute 2**20, then say the number.").text()) text += d;
assert.ok(text.replace(/,/g, "").includes("1048576"), text);
console.log(`[ok] streaming: ${text.trim().slice(0, 80)}`);

const session = new InMemorySession();
await run(chat, "My favourite colour is teal. Just say OK.", { session });
const r3 = await run(chat, "What is my favourite colour? One word.", { session });
assert.match(r3.output.toLowerCase(), /teal/);
console.log(`[ok] sessions: ${r3.output.trim()}`);
console.log("LIVE SMOKE TEST PASSED");
