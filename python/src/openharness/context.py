"""Run-scoped state: limits, cancellation, and the context passed to tools."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .types import Message, Usage

if TYPE_CHECKING:
    from .agent import Agent
    from .tracing import Trace

T = TypeVar("T")


@dataclass
class RunLimits:
    """Hard stops so an agent can never run forever.

    Every limit is optional. Defaults keep a runaway loop cheap: 20 model
    turns, 50 tool calls, 10 minutes.
    """

    max_turns: int | None = 20
    max_tool_calls: int | None = 50
    max_total_tokens: int | None = None
    timeout_seconds: float | None = 600
    max_output_retries: int = 2  # re-asks when structured output fails validation


class CancelToken:
    """Cooperative cancellation. Call ``cancel()`` from anywhere to stop a run.

    The run checks the token between steps and aborts in-flight model and tool
    calls as soon as it is cancelled.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()


@dataclass
class RunContext(Generic[T]):
    """Passed to tools, guardrails, dynamic instructions and hooks.

    ``deps`` is whatever object you pass to ``run(..., deps=...)``: a database
    handle, the current user, feature flags. It is never sent to the model.
    """

    agent: Agent
    deps: T | None = None
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turn: int = 0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)
    trace: Trace | None = None
    cancel_token: CancelToken | None = None
    state: dict[str, Any] = field(default_factory=dict)  # scratch space shared across the run

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at
