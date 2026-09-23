---
title: Testing
parent: Guides
nav_order: 12
---

# Testing agents

`FakeModel` replays a script of responses so you can test agent logic, tools, guardrails and limits with no network and
no API key. It records every request for assertions.

```python
from openharness import Agent, FakeModel, call, run

async def test_refund_flow():
    model = FakeModel([call("lookup_order", order_id="A1"), call("refund", order_id="A1", amount=20), "Done!"])
    result = await run(Agent(model=model, tools=[lookup_order, refund]), "Refund A1", approve=lambda c, t: True)
    assert result.output == "Done!"
    assert [m.name for m in result.new_messages if m.role == "tool"] == ["lookup_order", "refund"]
    assert "refund" in [t.name for t in model.requests[0].tools]
```

```ts
import { Agent, call, FakeModel, run } from "openharness";

const model = new FakeModel([call("lookup_order", { order_id: "A1" }), "Done!"]);
const result = await run(new Agent({ model, tools: [lookupOrder] }), "Refund A1");
```

Script steps can be:

* a string: the final answer
* `call(name, **args)`: one tool call
* `{"tool_calls": [...], "text": "..."}`: several calls or text plus calls
* `{"json": {...}}`: structured output
* a function of the request, for dynamic replies

When the script runs out the model repeats `default` / `defaultReply`, or echoes the last user message.
`OPENHARNESS_MODEL=fake:demo` gives you an echo model for smoke tests of whole apps.
