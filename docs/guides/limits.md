---
title: Limits, cancellation and approvals
parent: Guides
nav_order: 7
---

# Limits, cancellation and approvals
{: .no_toc }

1. TOC
{:toc}

## Limits

Every run is bounded. The defaults stop a runaway loop quickly and cheaply.

| Limit | Python | TypeScript | Default | Raises |
|---|---|---|---|---|
| Model turns | `max_turns` | `maxTurns` | 20 | `MaxTurnsExceeded` |
| Tool calls | `max_tool_calls` | `maxToolCalls` | 50 | `MaxToolCallsExceeded` |
| Tokens (input + output) | `max_total_tokens` | `maxTotalTokens` | none | `TokenBudgetExceeded` |
| Wall-clock time | `timeout_seconds` | `timeoutSeconds` | 600 | `RunTimeout` |
| Structured output retries | `max_output_retries` | `maxOutputRetries` | 2 | `OutputValidationError` |

Set them per agent or per run (per-run values win). Use `None` / `null` to turn a limit off.

```python
from openharness import RunLimits

agent = Agent(..., limits=RunLimits(max_turns=8, max_total_tokens=50_000))
result = await run(agent, "Research this", limits=RunLimits(timeout_seconds=30))
```

```ts
const agent = new Agent({ limits: { maxTurns: 8, maxTotalTokens: 50_000 } });
await run(agent, "Research this", { limits: { timeoutSeconds: 30 } });
```

All of these errors derive from `RunStopped` and carry `partial`, a `RunResult` with the messages, usage and turn
count so far:

```python
try:
    result = await run(agent, task)
except RunStopped as e:
    log.warning("stopped: %s after %d turns, %d tokens", e, e.partial.turns, e.partial.usage.total_tokens)
```

The timeout interrupts whatever is in flight, including a slow model call or a hung tool.

## Cancellation

Stop a run from anywhere: a UI button, a signal handler, another task.

```python
from openharness import CancelToken

token = CancelToken()
task = asyncio.create_task(run(agent, "Write a long report", cancel_token=token))
...
token.cancel("user clicked stop")        # the run raises RunCancelled
```

```ts
const token = new CancelToken();
const pending = run(agent, "Write a long report", { cancelToken: token });
token.cancel("user clicked stop");

// or a standard AbortSignal
const controller = new AbortController();
run(agent, "...", { signal: controller.signal });
controller.abort();
```

Streams have `stream.cancel()`. In TypeScript, tools receive `ctx.signal` so long operations (fetch, timers) can stop
immediately; in Python the tool's task is cancelled.

## Approvals (human in the loop)

Mark risky tools and decide per call with an `approve` callback. Without a callback, those tools are denied and the
model is told so.

```python
@tool(needs_approval=lambda args: args["amount"] > 100)
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""

async def approve(ctx, call) -> bool:
    return await ask_manager(call.name, call.arguments)

await run(agent, "Refund order A1 for $250", approve=approve)
```

```ts
const refund = tool({ name: "refund", description: "...", needsApproval: (args) => args.amount > 100, execute: ... });
await run(agent, "Refund order A1 for $250", { approve: async (ctx, call) => askManager(call.name, call.arguments) });
```

The CLI asks `Allow tool ...? [y/N]` in the terminal, or approves everything with `--yes`.
