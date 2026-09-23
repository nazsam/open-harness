---
title: Home
layout: home
nav_order: 1
---

# open-harness

Build AI agents that run inside your own app, on the model provider you choose, in **Python** or **TypeScript**.
{: .fs-6 .fw-300 }

[Get started](getting-started.html){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/nazsam/open-harness){: .btn .fs-5 .mb-4 .mb-md-0 }

---

open-harness is an open source SDK plus a ready-to-use harness. There is no hosted control plane: the agent loop,
your tools, your sessions and your traces all live in your process. Switch between OpenAI, Anthropic, Gemini and any
OpenAI-compatible server (Ollama, vLLM, Groq, OpenRouter and more) by changing one string.

## What you get

| Feature | What it means |
|---|---|
| **Harness** | `Harness()` gives you a working agent with tools, a conversation, limits and streaming in one line. |
| **Any provider** | `anthropic:claude-sonnet-5`, `openai:gpt-6-luna`, `gemini:gemini-3.8-flash`, `ollama:llama3.2` and more. |
| **Tools** | Plain functions become tools. Schemas come from type hints (Python) or `s.object(...)` (TypeScript). |
| **Structured output** | Typed, validated results with automatic retries when the model gets the shape wrong. |
| **Sessions** | Conversation history in memory, JSON Lines files or SQLite, or your own store. |
| **Memory** | Long-term facts across conversations, with `remember` and `recall` tools. |
| **Streaming** | Text deltas, tool calls, tool results and handoffs as events. |
| **Guardrails** | Input and output checks: length, patterns, PII redaction, or another agent as a judge. |
| **Limits** | Max turns, tool calls, tokens and wall-clock time, plus cancellation from anywhere. |
| **Approvals** | Mark risky tools `needs_approval` and decide per call. |
| **Tracing** | Spans for every model call, tool call, handoff and guardrail. Console, JSON Lines or OpenTelemetry. |
| **MCP** | Use any Model Context Protocol server over stdio or HTTP (2026-07-28 spec with legacy fallback). |
| **Multi-agent** | Handoffs between specialists and agents called as tools. |
| **CLI** | `openharness chat` in your terminal. |

## Thirty seconds

**Python**

```bash
pip install open-harness
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY, GEMINI_API_KEY
```

```python
from openharness import Agent, run_sync, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and cloudy in {city}"

agent = Agent(instructions="Be brief.", tools=[get_weather])
print(run_sync(agent, "Do I need a jacket in Toronto?").output)
```

**TypeScript**

```bash
npm install openharness
```

```ts
import { Agent, run, s, tool } from "openharness";

const getWeather = tool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: s.object({ city: s.string() }),
  execute: ({ city }) => `18C and cloudy in ${city}`,
});

const agent = new Agent({ instructions: "Be brief.", tools: [getWeather] });
console.log((await run(agent, "Do I need a jacket in Toronto?")).output);
```

**Terminal**

```bash
openharness chat --model anthropic:claude-sonnet-5
openharness chat --model ollama:llama3.2          # fully local
```

## Design principles

1. **Your process, your data.** No hosted service. Nothing leaves your app except the calls you make to your model provider.
2. **One loop, many providers.** Messages are stored in a neutral format, so a conversation started on one provider can continue on another.
3. **Safe by default.** Every run has turn, tool-call and time limits. Tools that need approval are denied unless you approve them.
4. **Same concepts in both languages.** The Python and TypeScript SDKs share names, behaviour and file formats. See [API parity](guides/parity.html).
5. **Small.** The Python SDK depends only on `httpx`. The TypeScript SDK has zero runtime dependencies.
