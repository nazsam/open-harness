// Test agent logic without any API key using FakeModel.
import assert from "node:assert/strict";
import { Agent, call, FakeModel, run, s, tool } from "openharness";

const lookupOrder = tool({
  name: "lookup_order",
  description: "Look up an order.",
  parameters: s.object({ order_id: s.string() }),
  execute: ({ order_id }) => ({ id: order_id, status: "shipped" }),
});

const model = new FakeModel([call("lookup_order", { order_id: "A123" }), "Your order A123 has shipped."]);
const result = await run(new Agent({ model, tools: [lookupOrder] }), "Where is order A123?");
assert.equal(result.output, "Your order A123 has shipped.");
assert.equal(result.newMessages[2].content, '{"id":"A123","status":"shipped"}');
console.log("ok:", result.output);
