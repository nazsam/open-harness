---
title: Getting started
nav_order: 2
---

# Getting started
{: .no_toc }

1. TOC
{:toc}

## Install

| | Command | Requires |
|---|---|---|
| Python | `pip install open-harness` | Python 3.10+ |
| TypeScript | `npm install openharness` | Node.js 20+ |

Both install an `openharness` command for the terminal.

{: .note }
Until the packages are published, install from source:
`pip install "git+https://github.com/nazsam/open-harness#subdirectory=python"` or clone the repo and run
`npm install && npm run build` inside `typescript/`.

## Pick a model

Set the API key for the provider you want. open-harness picks it up automatically.

```bash
export ANTHROPIC_API_KEY=...     # Claude
export OPENAI_API_KEY=...        # OpenAI
export GEMINI_API_KEY=...        # Gemini
```

Or choose a model explicitly with `provider:model`, in code or with `OPENHARNESS_MODEL`:

```bash
export OPENHARNESS_MODEL=ollama:llama3.2      # local, no key
```

See [Models and providers](guides/models.html) for the full list.

## Your first agent with the harness

The harness gives you a complete agent (calculator and clock tools, conversation memory, limits) in one line.

```python
from openharness import Harness

h = Harness()
print(h.ask_sync("What is 17.5% of 2,340?"))
print(h.ask_sync("And half of that?"))   # it remembers the conversation
```

```ts
import { Harness } from "openharness";

const h = new Harness();
console.log(await h.ask("What is 17.5% of 2,340?"));
console.log(await h.ask("And half of that?"));
```

Customise anything:

```python
h = Harness(
    "anthropic:claude-sonnet-5",
    instructions="You are a support agent for Acme.",
    tools=[lookup_order],
    session="chats/alice.jsonl",      # keep history on disk
    memory="data/memory.json",        # long-term facts
    limits=RunLimits(max_turns=8, timeout_seconds=60),
)
```

## Your first agent with the SDK

When you need full control, build an `Agent` and call `run`.

```python
import asyncio
from openharness import Agent, run, tool

@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID.

    Args:
        order_id: The order number, e.g. "A123".
    """
    return {"id": order_id, "status": "shipped", "eta": "Friday"}

agent = Agent(
    name="Support",
    instructions="Answer questions about orders. Be friendly and brief.",
    model="anthropic:claude-sonnet-5",
    tools=[lookup_order],
)

async def main():
    result = await run(agent, "Where is my order A123?")
    print(result.output)        # the answer
    print(result.usage)         # tokens used
    print(result.turns)         # model calls made

asyncio.run(main())
```

```ts
import { Agent, run, s, tool } from "openharness";

const lookupOrder = tool({
  name: "lookup_order",
  description: "Look up an order by its ID.",
  parameters: s.object({ order_id: s.string().describe('The order number, e.g. "A123".') }),
  execute: ({ order_id }) => ({ id: order_id, status: "shipped", eta: "Friday" }),
});

const agent = new Agent({
  name: "Support",
  instructions: "Answer questions about orders. Be friendly and brief.",
  model: "anthropic:claude-sonnet-5",
  tools: [lookupOrder],
});

const result = await run(agent, "Where is my order A123?");
console.log(result.output, result.usage.toJSON(), result.turns);
```

## How a run works

```
input ─► input guardrails ─► model ─┬─► tool calls? ─► run tools in parallel ─► back to model
                                     │
                                     └─► final answer ─► parse output type ─► output guardrails ─► result
```

Every turn checks the limits (turns, tool calls, tokens, time) and the cancel signal. A handoff tool switches the
active agent and the loop continues.

## Next steps

* [Tools](guides/tools.html)
* [Structured output](guides/structured-output.html)
* [Streaming](guides/streaming.html)
* [Examples](examples.html)
