"""Run results and the streaming handle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .types import Message, StreamEvent, Usage

if TYPE_CHECKING:
    from .agent import Agent
    from .context import CancelToken
    from .tracing import Trace


@dataclass
class RunResult:
    """What a run produced.

    ``output`` is the parsed structured output when the agent has an
    ``output_type``, otherwise the final text. ``new_messages`` holds every
    message created during the run (the input included).
    """

    output: Any
    text: str
    new_messages: list[Message]
    last_agent: Agent
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    tool_calls: int = 0
    trace: Trace | None = None
    history: list[Message] = field(default_factory=list)  # messages loaded from the session

    def to_messages(self) -> list[Message]:
        """Full conversation (history + this run), ready to pass as the next input."""
        return [*self.history, *self.new_messages]

    def __str__(self) -> str:
        return self.text


class RunStream:
    """Handle returned by ``run_stream``. Iterate for events, then await ``result()``.

    stream = run_stream(agent, "Tell me a story")
    async for event in stream:
        if event.type == "text_delta":
            print(event.data["delta"], end="")
    result = await stream.result()
    """

    def __init__(self, gen: AsyncIterator[StreamEvent], cancel_token: CancelToken):
        self._gen = gen
        self._cancel = cancel_token
        self._result: RunResult | None = None
        self._error: BaseException | None = None
        self._done = asyncio.Event()

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[StreamEvent]:
        try:
            async for event in self._gen:
                if event.type == "run_end":
                    self._result = event.data["result"]
                yield event
        except BaseException as e:
            self._error = e
            raise
        finally:
            self._done.set()

    async def text(self) -> AsyncIterator[str]:
        """Only the text deltas."""
        async for event in self:
            if event.type == "text_delta":
                yield event.data["delta"]

    async def result(self) -> RunResult:
        """Wait for the run to finish (consuming any remaining events)."""
        if not self._done.is_set():
            async for _ in self:
                pass
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancel.cancel(reason)
