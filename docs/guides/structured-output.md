---
title: Structured output
parent: Guides
nav_order: 3
---

# Structured output

Give an agent an `output_type` and `result.output` becomes a validated, typed object.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class Ticket:
    category: Literal["billing", "bug", "feature", "other"]
    priority: int
    summary: str
    customer_email: str | None = None

agent = Agent(instructions="Triage the support message.", output_type=Ticket)
ticket = (await run(agent, "I was charged twice!! sam@example.com")).output
print(ticket.category, ticket.priority)
```

```ts
const Ticket = s.object({
  category: s.enum(["billing", "bug", "feature", "other"]),
  priority: s.integer(),
  summary: s.string(),
  customer_email: s.string().optional(),
});

const agent = new Agent({ instructions: "Triage the support message.", outputType: Ticket });
const ticket = (await run(agent, "I was charged twice!! sam@example.com")).output; // typed as Infer<typeof Ticket>
```

Python accepts dataclasses, Pydantic models, `TypedDict`, lists, primitives, enums and raw JSON Schema dicts.
Non-object types (for example `list[str]`) are wrapped in `{"value": ...}` for the model and unwrapped for you.

## How it works

1. The schema is sent using each provider's native structured output feature: `response_format` (OpenAI, strict mode
   when the schema allows it), `output_config.format` (Anthropic) and `responseJsonSchema` (Gemini).
2. The reply is parsed (code fences and surrounding prose are tolerated) and validated.
3. If validation fails, the error is sent back and the model tries again, up to `RunLimits.max_output_retries`
   (default 2). After that the run raises `OutputValidationError`.

Tools and structured output work together: the agent can call tools first and give the typed answer at the end.

{: .note }
Gemini does not combine a response schema with function calling in one request, so when a Gemini agent has tools the
schema is given in the instructions instead and still validated on return.
