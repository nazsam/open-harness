---
title: Tracing
parent: Guides
nav_order: 8
---

# Tracing

Every run produces a trace: a tree of spans for the run, each model call, each tool call, handoffs, guardrails and MCP
tool listing. Spans carry timings, token usage, arguments, outputs (truncated) and errors.

```
[trace]     model:anthropic:claude-sonnet-5 812.4ms in=412 out=38
[trace]       tool:get_weather 3.1ms
[trace]     model:anthropic:claude-sonnet-5 640.2ms in=498 out=51
[trace]   run:Travel helper 1460.9ms
```

## Turning it on

With no code changes:

```bash
OPENHARNESS_TRACE=console python app.py          # print spans to stderr
OPENHARNESS_TRACE=./traces.jsonl node app.js     # one JSON trace per line
```

Per run, or for the whole process:

```python
from openharness import ConsoleProcessor, JSONLProcessor, add_trace_processor

await run(agent, "hi", trace=ConsoleProcessor())
add_trace_processor(JSONLProcessor("traces.jsonl"))
```

```ts
import { addTraceProcessor, ConsoleProcessor, JSONLProcessor } from "openharness";

await run(agent, "hi", { trace: new ConsoleProcessor() });
addTraceProcessor(new JSONLProcessor("traces.jsonl"));
```

`result.trace` also holds the trace object, and `MemoryProcessor` collects traces in a list (handy in tests).

## OpenTelemetry

In Python, `OpenTelemetryProcessor` re-emits spans through `opentelemetry-api`, so they flow to Jaeger, Honeycomb,
Datadog, Langfuse or any OTel backend you already run:

```bash
pip install "open-harness[otel]"
```

```python
from openharness import OpenTelemetryProcessor, add_trace_processor
add_trace_processor(OpenTelemetryProcessor())
```

## Your own processor

Implement any of `on_span_start`, `on_span_end`, `on_trace_end` (`onSpanStart`, `onSpanEnd`, `onTraceEnd`):

```python
class CostTracker:
    def on_span_start(self, span): pass
    def on_span_end(self, span):
        if span.kind == "model":
            metrics.increment("llm.tokens", span.attributes["usage"]["total_tokens"])
    def on_trace_end(self, trace): pass
```

A failing processor never breaks a run; the error is printed and the run continues.
