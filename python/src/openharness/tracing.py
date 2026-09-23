"""Tracing: see every model call, tool call, handoff and guardrail in a run.

Each run creates a ``Trace`` made of nested ``Span``s. Finished traces go to
processors. Built-in processors print to the console or append JSON Lines to
a file; ``OpenTelemetryProcessor`` forwards spans to any OTel backend when
``opentelemetry-api`` is installed.

    run(agent, "hi", trace=ConsoleProcessor())
    # or globally:
    add_trace_processor(JSONLProcessor("traces.jsonl"))
    # or with no code change:
    OPENHARNESS_TRACE=console | OPENHARNESS_TRACE=./traces.jsonl
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

from .types import new_id


@dataclass
class Span:
    kind: str  # run | agent | model | tool | guardrail | handoff | mcp
    name: str
    trace_id: str
    id: str = field(default_factory=lambda: new_id("span"))
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        return None if self.end is None else round((self.end - self.start) * 1000, 1)

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "error": self.error,
        }


class TraceProcessor(Protocol):
    def on_span_start(self, span: Span) -> None: ...
    def on_span_end(self, span: Span) -> None: ...
    def on_trace_end(self, trace: Trace) -> None: ...


class Trace:
    def __init__(
        self, name: str, processors: list[TraceProcessor] | None = None, metadata: dict[str, Any] | None = None
    ):
        self.id = new_id("trace")
        self.name = name
        self.metadata = metadata or {}
        self.spans: list[Span] = []
        self.processors = processors or []
        self._stack: list[Span] = []

    @contextlib.contextmanager
    def span(self, kind: str, name: str, parent: Span | None = None, **attrs: Any) -> Iterator[Span]:
        """Open a span. Pass ``parent`` for work that runs concurrently (it is then kept off the stack)."""
        concurrent = parent is not None
        parent_id = parent.id if parent else (self._stack[-1].id if self._stack else None)
        sp = Span(kind=kind, name=name, trace_id=self.id, parent_id=parent_id, attributes=dict(attrs))
        self.spans.append(sp)
        if not concurrent:
            self._stack.append(sp)
        self._emit("on_span_start", sp)
        try:
            yield sp
        except BaseException as e:
            sp.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            sp.end = time.time()
            if sp in self._stack:
                self._stack.remove(sp)
            self._emit("on_span_end", sp)

    def finish(self) -> None:
        self._emit("on_trace_end", self)

    def _emit(self, method: str, arg: Any) -> None:
        for p in self.processors:
            try:
                getattr(p, method)(arg)
            except Exception as e:  # never let tracing break a run
                print(f"[openharness] trace processor {type(p).__name__} failed: {e}", file=sys.stderr)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "metadata": self.metadata, "spans": [s.to_dict() for s in self.spans]}


# ------------------------------------------------------------------ processors


class ConsoleProcessor:
    """Prints one line per finished span, indented by depth."""

    def __init__(self, stream: TextIO | None = None, show_attributes: bool = False):
        self.stream = stream or sys.stderr
        self.show_attributes = show_attributes
        self._depth: dict[str, int] = {}

    def on_span_start(self, span: Span) -> None:
        self._depth[span.id] = self._depth.get(span.parent_id or "", -1) + 1

    def on_span_end(self, span: Span) -> None:
        pad = "  " * self._depth.pop(span.id, 0)
        status = f" ERROR {span.error}" if span.error else ""
        extra = ""
        a = span.attributes
        if span.kind == "model" and "usage" in a:
            u = a["usage"]
            extra = f" in={u.get('input_tokens')} out={u.get('output_tokens')}"
        if self.show_attributes and a:
            extra += " " + json.dumps(a, default=str)[:300]
        print(f"[trace] {pad}{span.kind}:{span.name} {span.duration_ms}ms{extra}{status}", file=self.stream)

    def on_trace_end(self, trace: Trace) -> None:
        return None


class JSONLProcessor:
    """Appends each finished trace as one JSON line."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def on_span_start(self, span: Span) -> None:
        return None

    def on_span_end(self, span: Span) -> None:
        return None

    def on_trace_end(self, trace: Trace) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), default=str) + "\n")


class MemoryProcessor:
    """Keeps finished traces in a list. Handy for tests and dashboards."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def on_span_start(self, span: Span) -> None:
        return None

    def on_span_end(self, span: Span) -> None:
        return None

    def on_trace_end(self, trace: Trace) -> None:
        self.traces.append(trace)


class OpenTelemetryProcessor:
    """Re-emits spans through OpenTelemetry. Requires ``opentelemetry-api``."""

    def __init__(self, tracer_name: str = "openharness"):
        from opentelemetry import trace as otel  # type: ignore[import-not-found]

        self._otel = otel
        self._tracer = otel.get_tracer(tracer_name)
        self._live: dict[str, Any] = {}

    def on_span_start(self, span: Span) -> None:
        parent = self._live.get(span.parent_id or "")
        ctx = self._otel.set_span_in_context(parent) if parent is not None else None
        self._live[span.id] = self._tracer.start_span(
            f"{span.kind} {span.name}", context=ctx, start_time=int(span.start * 1e9)
        )

    def on_span_end(self, span: Span) -> None:
        s = self._live.pop(span.id, None)
        if s is None:
            return
        for k, v in span.attributes.items():
            s.set_attribute(
                f"openharness.{k}", v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)
            )
        if span.error:
            s.set_attribute("error", span.error)
        s.end(end_time=int((span.end or time.time()) * 1e9))

    def on_trace_end(self, trace: Trace) -> None:
        return None


_GLOBAL: list[TraceProcessor] = []


def add_trace_processor(processor: TraceProcessor) -> None:
    _GLOBAL.append(processor)


def clear_trace_processors() -> None:
    _GLOBAL.clear()


def processors_from_env() -> list[TraceProcessor]:
    value = os.environ.get("OPENHARNESS_TRACE", "").strip()
    if not value or value.lower() in ("0", "false", "off", "none"):
        return []
    if value.lower() in ("1", "true", "console", "stderr"):
        return [ConsoleProcessor()]
    return [JSONLProcessor(value)]


def global_processors() -> list[TraceProcessor]:
    return [*_GLOBAL, *processors_from_env()]
