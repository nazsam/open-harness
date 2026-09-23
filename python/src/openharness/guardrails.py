"""Guardrails check input before the model sees it and output before your app does.

A guardrail is a function ``(ctx, value) -> GuardrailResult``. Set
``tripwire=True`` to stop the run with ``GuardrailTripped``, or return
``replacement`` to rewrite the value (for example to redact PII) and carry on.

    @input_guardrail
    def no_homework(ctx, text):
        return GuardrailResult(tripwire="solve my homework" in text.lower(), info="homework")

    agent = Agent(..., input_guardrails=[no_homework, max_length(4000)])
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import Agent
    from .context import RunContext


@dataclass
class GuardrailResult:
    tripwire: bool = False
    info: Any = None
    replacement: Any = None  # when set (and not tripped) the value is replaced

    @classmethod
    def ok(cls, info: Any = None) -> GuardrailResult:
        return cls(False, info)

    @classmethod
    def block(cls, info: Any = None) -> GuardrailResult:
        return cls(True, info)


GuardFn = Callable[["RunContext", Any], "GuardrailResult | bool | Awaitable[GuardrailResult | bool]"]


@dataclass
class Guardrail:
    fn: GuardFn
    name: str
    stage: str  # "input" | "output"

    async def check(self, ctx: RunContext, value: Any) -> GuardrailResult:
        result = self.fn(ctx, value)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, bool):
            return GuardrailResult(tripwire=result)
        if result is None:
            return GuardrailResult()
        return result


def input_guardrail(fn: GuardFn | None = None, *, name: str | None = None) -> Any:
    def wrap(f: GuardFn) -> Guardrail:
        return Guardrail(f, name or f.__name__, "input")

    return wrap(fn) if fn else wrap


def output_guardrail(fn: GuardFn | None = None, *, name: str | None = None) -> Any:
    def wrap(f: GuardFn) -> Guardrail:
        return Guardrail(f, name or f.__name__, "output")

    return wrap(fn) if fn else wrap


# ------------------------------------------------------------------ built-ins


def max_length(chars: int, stage: str = "input") -> Guardrail:
    """Block text longer than ``chars`` characters."""

    def check(_ctx: RunContext, value: Any) -> GuardrailResult:
        n = len(str(value))
        return GuardrailResult(n > chars, {"length": n, "limit": chars})

    return Guardrail(check, f"max_length({chars})", stage)


def blocked_patterns(patterns: list[str], stage: str = "input", flags: int = re.IGNORECASE) -> Guardrail:
    """Block text matching any regular expression."""
    compiled = [re.compile(p, flags) for p in patterns]

    def check(_ctx: RunContext, value: Any) -> GuardrailResult:
        text = str(value)
        hits = [p.pattern for p in compiled if p.search(text)]
        return GuardrailResult(bool(hits), {"matched": hits})

    return Guardrail(check, "blocked_patterns", stage)


PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)",
    "credit_card": r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)",
    "ssn": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
    "ip_address": r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)",
}


def pii(action: str = "redact", kinds: list[str] | None = None, stage: str = "input") -> Guardrail:
    """Detect common PII. ``action="redact"`` masks it, ``"block"`` stops the run."""
    selected = {k: re.compile(v) for k, v in PII_PATTERNS.items() if kinds is None or k in kinds}

    def check(_ctx: RunContext, value: Any) -> GuardrailResult:
        if not isinstance(value, str):
            return GuardrailResult()
        found: dict[str, int] = {}
        text = value
        for kind, pattern in selected.items():
            text, n = pattern.subn(f"[{kind.upper()}]", text)
            if n:
                found[kind] = n
        if not found:
            return GuardrailResult()
        if action == "block":
            return GuardrailResult(True, {"found": found})
        return GuardrailResult(False, {"found": found}, replacement=text)

    return Guardrail(check, f"pii({action})", stage)


def llm_guardrail(agent: Agent, *, stage: str = "input", name: str = "llm_guardrail") -> Guardrail:
    """Use another agent as a classifier.

    The agent must have ``output_type`` with a boolean ``tripwire`` field
    (and optionally ``reason``). It is run on the text being checked.
    """
    from .runner import run

    async def check(ctx: RunContext, value: Any) -> GuardrailResult:
        result = await run(agent, str(value), deps=ctx.deps)
        out = result.output
        tripped = out.get("tripwire") if isinstance(out, dict) else getattr(out, "tripwire", False)
        return GuardrailResult(bool(tripped), out)

    return Guardrail(check, name, stage)
