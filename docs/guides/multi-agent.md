---
title: Multi-agent
parent: Guides
nav_order: 10
---

# Multi-agent

Two patterns cover most multi-agent systems, and they combine freely.

## Handoffs: pass the conversation to a specialist

The model sees a `transfer_to_<agent>` tool for each handoff. When it calls one, the target agent takes over the
conversation and answers the user directly. `result.last_agent` tells you who finished.

```python
billing = Agent(name="Billing", description="Refunds, invoices and payments.", instructions="...")
tech = Agent(name="Tech support", description="Bugs and setup help.", instructions="...")
triage = Agent(name="Triage", instructions="Route the user to the right specialist.", handoffs=[billing, tech])

result = await run(triage, "I was charged twice this month")
print(result.last_agent.name)   # "Billing"
```

```ts
const triage = new Agent({ name: "Triage", instructions: "Route the user to the right specialist.", handoffs: [billing, tech] });
const result = await run(triage, "I was charged twice this month");
console.log(result.lastAgent.name);
```

Customise with `Handoff`:

```python
from openharness import Handoff

Handoff(billing, tool_name="escalate_to_billing", description="Only for payment problems",
        input_filter=lambda history: history[-4:])   # pass only recent messages
```

Specialists can hand back (`billing.handoffs = [triage]`). The run's `max_turns` limit still applies across handoffs.

## Agents as tools: a manager with workers

`agent.as_tool()` turns an agent into a tool. The manager stays in control, calls workers (in parallel when it asks
for several at once) and combines their answers. Worker token usage is added to the run's usage, and worker runs appear
nested in the trace.

```python
researcher = Agent(name="Researcher", instructions="Give 3 factual bullet points.", model="gemini:gemini-3.8-flash")
writer = Agent(name="Writer", instructions="Write clear, short prose.", model="anthropic:claude-sonnet-5")
manager = Agent(
    name="Manager",
    instructions="Research first, then have the writer draft the answer.",
    tools=[researcher.as_tool(), writer.as_tool(description="Turns notes into a polished answer")],
)
```

```ts
const manager = new Agent({ name: "Manager", tools: [researcher.asTool(), writer.asTool({ description: "Turns notes into a polished answer" })] });
```

Workers can use different providers, which is an easy way to mix a cheap fast model with a stronger one.

## Which to use

| | Handoff | Agent as tool |
|---|---|---|
| Who talks to the user at the end | The specialist | The manager |
| Conversation history | Shared | Worker only sees its input |
| Good for | Routing, triage, escalation | Research, drafting, review, fan-out |
