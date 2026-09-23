---
title: API parity
parent: Guides
nav_order: 13
---

# Python and TypeScript side by side

The SDKs share concepts, behaviour, defaults and file formats. Names follow each language's conventions.

| Concept | Python | TypeScript |
|---|---|---|
| Create an agent | `Agent(name=..., instructions=..., tools=[...])` | `new Agent({ name, instructions, tools })` |
| Run | `await run(agent, input)` | `await run(agent, input)` |
| Run (blocking) | `run_sync(agent, input)` | n/a (top-level `await`) |
| Stream | `run_stream(agent, input)` | `runStream(agent, input)` |
| Harness | `Harness(model).ask_sync(...)` / `await h.ask(...)` | `await new Harness(model).ask(...)` |
| Define a tool | `@tool` decorator on a function | `tool({ name, description, parameters, execute })` |
| Tool schema | Type hints and docstring | `s.object({...})` |
| Structured output | `output_type=MyDataclass` | `outputType: s.object({...})` |
| Your objects | `deps=...`, `ctx.deps` | `{ deps }`, `ctx.deps` |
| Limits | `RunLimits(max_turns=8)` | `{ maxTurns: 8 }` |
| Cancel | `CancelToken().cancel()` | `CancelToken` or `AbortSignal` |
| Approvals | `approve=fn(ctx, call)` | `{ approve: (ctx, call) => ... }` |
| Sessions | `InMemorySession`, `FileSession`, `SQLiteSession` | `InMemorySession`, `FileSession` |
| Memory | `InMemoryMemory`, `FileMemory` | `InMemoryMemory`, `FileMemory` |
| Guardrails | `@input_guardrail`, `max_length`, `pii()` | `inputGuardrail(name, fn)`, `maxLength`, `pii()` |
| Tracing | `ConsoleProcessor`, `JSONLProcessor`, `OpenTelemetryProcessor` | `ConsoleProcessor`, `JSONLProcessor` |
| MCP | `MCPServerStdio`, `MCPServerHTTP` | `MCPServerStdio`, `MCPServerHTTP` |
| Handoffs | `handoffs=[agent]` | `handoffs: [agent]` |
| Agent as tool | `agent.as_tool()` | `agent.asTool()` |
| Testing | `FakeModel([call("x", a=1), "done"])` | `new FakeModel([call("x", { a: 1 }), "done"])` |
| Result | `result.output`, `.usage`, `.turns`, `.new_messages`, `.last_agent` | `result.output`, `.usage`, `.turns`, `.newMessages`, `.lastAgent` |

**Shared on disk:** `FileSession` JSON Lines and `FileMemory` JSON use the same format, so both SDKs can read and write
the same files. **Shared in the environment:** `OPENHARNESS_MODEL`, `OPENHARNESS_TRACE`, provider API keys and
`<PROVIDER>_BASE_URL`.
