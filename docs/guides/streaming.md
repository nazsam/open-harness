---
title: Streaming
parent: Guides
nav_order: 4
---

# Streaming

`run_stream` / `runStream` yields events while the agent works. Await `result()` at the end for the full `RunResult`.

```python
from openharness import run_stream

stream = run_stream(agent, "Plan a 3 day trip to Lisbon")
async for event in stream:
    match event.type:
        case "text_delta":  print(event.data["delta"], end="", flush=True)
        case "tool_call":   print(f"\n-> {event.data['name']}({event.data['arguments']})")
        case "tool_result": print(f"<- {event.data['output'][:80]}")
        case "handoff":     print(f"\n[{event.data['from']} -> {event.data['to']}]")
result = await stream.result()
```

```ts
const stream = runStream(agent, "Plan a 3 day trip to Lisbon");
for await (const event of stream) {
  if (event.type === "text_delta") process.stdout.write(event.data.delta);
  else if (event.type === "tool_call") console.log(`\n-> ${event.data.name}`, event.data.arguments);
}
const result = await stream.result();
```

Only want text? `async for chunk in stream.text()` / `for await (const chunk of stream.text())`.

## Event types

| Type | Data |
|---|---|
| `agent_start` | `agent` |
| `text_delta` | `delta` |
| `tool_call` | `id`, `name`, `arguments` |
| `tool_result` | `id`, `name`, `output`, `is_error` (`isError` in TypeScript) |
| `handoff` | `from`, `to` |
| `message` | `message` (the full assistant message for a turn) |
| `run_end` | `result` |

Every event also carries `agent` (the active agent's name) and `timestamp`.

## Stopping a stream

Call `stream.cancel()` at any time. The in-flight model request or tool call is aborted and the run raises
`RunCancelled`.

## Callbacks instead of iteration

```python
result = await run(agent, "hi", on_event=lambda e: print(e.type))
```

```ts
const result = await run(agent, "hi", { onEvent: (e) => console.log(e.type) });
```
