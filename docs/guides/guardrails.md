---
title: Guardrails
parent: Guides
nav_order: 6
---

# Guardrails

Input guardrails run on the user's message before the model sees it. Output guardrails run on the final output before
your app gets it. A guardrail can:

* **block**: return `tripwire=True`; the run raises `GuardrailTripped` with `guardrail`, `stage` and `info`.
* **rewrite**: return a `replacement` value (for example redacted text) and let the run continue.
* **pass**: return `False` or nothing.

## Built-in guardrails

| Guardrail | Python | TypeScript |
|---|---|---|
| Length limit | `max_length(4000)` | `maxLength(4000)` |
| Regex block list | `blocked_patterns([r"ignore previous instructions"])` | `blockedPatterns([/ignore previous instructions/i])` |
| PII redact or block | `pii()`, `pii("block", kinds=["credit_card"])` | `pii()`, `pii({ action: "block", kinds: ["credit_card"] })` |
| Another agent as judge | `llm_guardrail(judge_agent)` | `llmGuardrail(judgeAgent)` |

PII covers emails, phone numbers, card numbers, US SSNs and IP addresses; redaction replaces them with `[EMAIL]`,
`[PHONE]` and so on.

## Custom guardrails

```python
from openharness import GuardrailResult, input_guardrail, output_guardrail

@input_guardrail
def on_topic(ctx, text: str):
    return GuardrailResult(tripwire="crypto" in text.lower(), info="off topic")

@output_guardrail
async def no_prices(ctx, output):
    return "$" in str(output)     # a bool is fine too

agent = Agent(input_guardrails=[on_topic, max_length(4000), pii()], output_guardrails=[no_prices])
```

```ts
import { inputGuardrail, outputGuardrail } from "openharness";

const onTopic = inputGuardrail("on_topic", (_ctx, text) => ({ tripwire: text.toLowerCase().includes("crypto"), info: "off topic" }));
const noPrices = outputGuardrail("no_prices", (_ctx, output) => String(output).includes("$"));
const agent = new Agent({ inputGuardrails: [onTopic], outputGuardrails: [noPrices] });
```

## Handling a tripped guardrail

```python
try:
    result = await run(agent, user_text)
except GuardrailTripped as e:
    reply = "Sorry, I can't help with that." if e.stage == "input" else "Let me rephrase that."
```

## A model as the judge

```python
@dataclass
class Verdict:
    tripwire: bool
    reason: str

judge = Agent(name="Judge", model="openai:gpt-6-luna", output_type=Verdict,
              instructions="Set tripwire=true if the message asks for medical dosing advice.")
agent = Agent(input_guardrails=[llm_guardrail(judge)])
```
