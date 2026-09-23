---
title: Tools
parent: Guides
nav_order: 2
---

# Tools
{: .no_toc }

1. TOC
{:toc}

## Defining tools

**Python**: decorate a function. The schema comes from type hints, the description from the docstring, and argument
descriptions from the `Args:` section. Sync and async functions both work (sync ones run in a thread).

```python
from typing import Literal
from openharness import tool

@tool
def search_products(query: str, category: Literal["books", "games"] | None = None, limit: int = 5) -> list[dict]:
    """Search the product catalogue.

    Args:
        query: Words to search for.
        category: Optional category filter.
        limit: Maximum results.
    """
    return db.search(query, category, limit)
```

Supported types: `str`, `int`, `float`, `bool`, `list[...]`, `dict[str, ...]`, `Literal`, `Enum`, `Optional`/unions,
dataclasses, `TypedDict` and Pydantic models. Plain functions passed in `tools=[...]` are converted automatically.

**TypeScript**: use `tool()` with an `s.object(...)` schema; argument types are inferred.

```ts
import { s, tool } from "openharness";

const searchProducts = tool({
  name: "search_products",
  description: "Search the product catalogue.",
  parameters: s.object({
    query: s.string().describe("Words to search for."),
    category: s.enum(["books", "games"]).optional(),
    limit: s.integer().default(5),
  }),
  execute: async ({ query, category, limit }) => db.search(query, category, limit),
});
```

The schema builder `s` has `string`, `number`, `integer`, `boolean`, `enum`, `literal`, `array`, `object`, `record`,
`union`, `any` and `json` (wrap an existing JSON Schema), with `.describe()`, `.optional()`, `.nullable()` and `.default()`.

## Run context and dependencies

Tools can receive the run context. Put your own objects (database handles, the current user) in `deps`; they reach
tools but are never sent to the model.

```python
from openharness import RunContext, tool

@tool
async def my_orders(ctx: RunContext) -> list[dict]:
    """List the current user's orders."""
    return await ctx.deps.db.orders_for(ctx.deps.user_id)

await run(agent, "What did I order?", deps=AppDeps(db=db, user_id="u_42"))
```

```ts
const myOrders = tool({
  name: "my_orders",
  description: "List the current user's orders.",
  execute: (_args, ctx) => ctx.deps.db.ordersFor(ctx.deps.userId),
});
await run(agent, "What did I order?", { deps: { db, userId: "u_42" } });
```

The context also exposes `usage`, `turn`, `tool_calls`, `messages`, `state` (a scratch dict for the run) and, in
TypeScript, `signal`, which aborts when the run is cancelled or times out.

## Errors

Anything a tool raises is sent back to the model as an error result, so the model can recover. Raise `ToolError` for a
clean message. Arguments are validated against the schema before your function runs.

```python
from openharness import ToolError

@tool
def transfer(amount: float) -> str:
    """Transfer money."""
    if amount > 1000:
        raise ToolError("Transfers over 1000 need manual review")
    ...
```

## Timeouts and approvals

```python
@tool(timeout=10, needs_approval=True)
def delete_customer(customer_id: str) -> str:
    """Permanently delete a customer."""
```

```ts
const deleteCustomer = tool({ name: "delete_customer", description: "...", timeoutSeconds: 10, needsApproval: true, execute: ... });
```

`needs_approval` can also be a function of the arguments. See [Limits, cancellation and approvals](limits.html).

## Parallel tool calls

When the model asks for several tools in one turn, they run concurrently and results are returned in order.

## Built-in tools

| Tool | What it does |
|---|---|
| `calculator` | Safe arithmetic and math functions, no `eval`. |
| `current_time` | Current date and time in any IANA time zone. |
| `fetch_url` | Download a page as text. Requires approval. |

```python
from openharness import calculator, current_time
agent = Agent(tools=[calculator, current_time])
```
