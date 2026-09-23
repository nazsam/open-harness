# open-harness

**Build AI agents that run inside your own app, on the model provider you choose, in Python or TypeScript.**

[![CI](https://github.com/nazsam/open-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/nazsam/open-harness/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-nazsam.github.io%2Fopen--harness-blue)](https://nazsam.github.io/open-harness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

open-harness is an open source agent SDK plus a ready-to-use harness. There is no hosted control plane: the agent
loop, your tools, sessions, memory and traces all live in your process. Switch between OpenAI, Anthropic, Gemini and
any OpenAI-compatible server (Ollama, vLLM, LM Studio, Groq, OpenRouter and more) by changing one string.

```python
from openharness import Harness

h = Harness("anthropic:claude-sonnet-5")          # or "openai:gpt-6-luna", "gemini:gemini-3.8-flash", "ollama:llama3.2"
print(h.ask_sync("What is 17.5% of 2,340?"))       # tools, conversation, limits and streaming are already wired up
```

```bash
openharness chat --model ollama:llama3.2          # chat from the terminal
```

## Features

| | |
|---|---|
| **Harness** | A working agent in one line: tools, conversation, limits, streaming, terminal chat. |
| **Any provider** | Anthropic, OpenAI, Gemini, and 9 OpenAI-compatible presets. Conversations can even move between providers. |
| **Tools** | Functions become tools. Schemas from Python type hints or the TypeScript `s` schema builder. Parallel execution. |
| **Structured output** | Typed, validated results using each provider's native JSON schema mode, with automatic retries. |
| **Sessions** | Conversation history in memory, JSON Lines or SQLite, or your own store. Same file format in both languages. |
| **Memory** | Long-term facts across conversations with `remember` / `recall` tools and automatic recall. |
| **Streaming** | Text deltas, tool calls, tool results and handoffs as events. Cancel mid-reply. |
| **Guardrails** | Length, regex, PII redaction, or another agent as a judge, on input and output. |
| **Limits** | Max turns, tool calls, tokens and wall-clock time. Cancel from anywhere. Nothing runs forever. |
| **Approvals** | Human-in-the-loop for risky tools. |
| **Tracing** | Spans for every model call, tool call, handoff and guardrail: console, JSON Lines, OpenTelemetry. |
| **MCP** | stdio and HTTP servers, MCP 2026-07-28 with fallback to older handshake-based servers. |
| **Multi-agent** | Handoffs between specialists and agents used as tools. |
| **Small** | Python depends only on `httpx`. TypeScript has zero runtime dependencies. |

## Install

```bash
pip install open-harness        # Python 3.10+
npm install openharness         # Node.js 20+
```

Until the first release is published, install from this repo:

```bash
pip install "git+https://github.com/nazsam/open-harness#subdirectory=python"
git clone https://github.com/nazsam/open-harness && cd open-harness/typescript && npm install && npm run build
```

## A real agent in both languages

**Python**

```python
import asyncio
from dataclasses import dataclass
from openharness import Agent, RunLimits, pii, run, tool

@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order.

    Args:
        order_id: The order number, e.g. "A123".
    """
    return {"id": order_id, "status": "shipped", "eta": "Friday"}

@dataclass
class Answer:
    reply: str
    order_status: str | None

agent = Agent(
    name="Support",
    instructions="Help customers with their orders.",
    model="anthropic:claude-sonnet-5",
    tools=[lookup_order],
    output_type=Answer,
    input_guardrails=[pii()],
    limits=RunLimits(max_turns=6, timeout_seconds=60),
)

result = asyncio.run(run(agent, "Where is order A123? My email is sam@example.com"))
print(result.output.reply, result.usage)
```

**TypeScript**

```ts
import { Agent, pii, run, s, tool } from "openharness";

const lookupOrder = tool({
  name: "lookup_order",
  description: "Look up an order.",
  parameters: s.object({ order_id: s.string().describe('The order number, e.g. "A123".') }),
  execute: ({ order_id }) => ({ id: order_id, status: "shipped", eta: "Friday" }),
});

const agent = new Agent({
  name: "Support",
  instructions: "Help customers with their orders.",
  model: "anthropic:claude-sonnet-5",
  tools: [lookupOrder],
  outputType: s.object({ reply: s.string(), order_status: s.string().nullable() }),
  inputGuardrails: [pii()],
  limits: { maxTurns: 6, timeoutSeconds: 60 },
});

const result = await run(agent, "Where is order A123? My email is sam@example.com");
console.log(result.output.reply, result.usage.toJSON());
```

## Documentation

**[nazsam.github.io/open-harness](https://nazsam.github.io/open-harness/)**

[Getting started](https://nazsam.github.io/open-harness/getting-started.html) ·
[Models](https://nazsam.github.io/open-harness/guides/models.html) ·
[Tools](https://nazsam.github.io/open-harness/guides/tools.html) ·
[Structured output](https://nazsam.github.io/open-harness/guides/structured-output.html) ·
[Streaming](https://nazsam.github.io/open-harness/guides/streaming.html) ·
[Sessions and memory](https://nazsam.github.io/open-harness/guides/sessions-and-memory.html) ·
[Guardrails](https://nazsam.github.io/open-harness/guides/guardrails.html) ·
[Limits](https://nazsam.github.io/open-harness/guides/limits.html) ·
[Tracing](https://nazsam.github.io/open-harness/guides/tracing.html) ·
[MCP](https://nazsam.github.io/open-harness/guides/mcp.html) ·
[Multi-agent](https://nazsam.github.io/open-harness/guides/multi-agent.html) ·
[CLI](https://nazsam.github.io/open-harness/guides/cli.html) ·
[Examples](https://nazsam.github.io/open-harness/examples.html)

## Repository layout

```
python/          Python SDK (package: open-harness, import: openharness)
typescript/      TypeScript SDK (package: openharness)
examples/        The same 9 examples in both languages
docs/            Documentation site (GitHub Pages)
```

## Development

```bash
cd python && pip install -e ".[dev]" && pytest -q && ruff check src tests
cd typescript && npm install && npm test
```

Tests run offline: providers are exercised against recorded-style HTTP responses, MCP against local test servers,
and agent logic with the scripted `FakeModel`.

## License

[MIT](LICENSE) (c) 2026 Sam Naz
