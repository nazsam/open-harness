"""A scripted model for tests and offline demos. No network, no API key."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..types import Message, ModelRequest, ModelResponse, ToolCall, Usage, new_id
from .base import Model, ModelEvent

Step = str | dict[str, Any] | Message | Callable[[ModelRequest], "str | dict[str, Any] | Message"]


def call(name: str, **arguments: Any) -> dict[str, Any]:
    """Script a tool call: ``FakeModel([call("add", a=1, b=2), "The answer is 3"])``."""
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


class FakeModel(Model):
    """Replays a script of responses, one per model call.

    Each step is a string (final text), ``call(...)`` / a dict with
    ``text`` and ``tool_calls``, a ``Message``, or a function of the request.
    When the script runs out it repeats ``default`` (or echoes the last user
    message). Every request is recorded in ``requests`` for assertions.
    """

    provider = "fake"

    def __init__(
        self,
        script: list[Step] | None = None,
        *,
        model: str = "fake-model",
        default: str | None = None,
        chunk_size: int = 4,
    ):
        self.model = model
        self.script = list(script or [])
        self.default = default
        self.chunk_size = chunk_size
        self.requests: list[ModelRequest] = []

    def _next(self, req: ModelRequest) -> Message:
        if self.script:
            step: Any = self.script.pop(0)
        elif self.default is not None:
            step = self.default
        else:
            last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            step = f"echo: {last}"
        if callable(step) and not isinstance(step, Message):
            step = step(req)
        if isinstance(step, Message):
            return step
        if isinstance(step, str):
            return Message.assistant(step)
        calls = [
            ToolCall(id=tc.get("id") or new_id("call"), name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in step.get("tool_calls", [])
        ]
        text = step.get("text", "")
        if "json" in step:
            text = json.dumps(step["json"])
        return Message.assistant(text, calls)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        msg = self._next(request)
        tokens = max(1, len(msg.content) // 4)
        return ModelResponse(
            msg, Usage(input_tokens=10, output_tokens=tokens, requests=1), "tool_use" if msg.tool_calls else "stop"
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        response = await self.generate(request)
        text = response.message.content
        for i in range(0, len(text), self.chunk_size):
            yield ModelEvent("text_delta", {"delta": text[i : i + self.chunk_size]})
        for tc in response.message.tool_calls:
            yield ModelEvent("tool_call", tc.to_dict())
        yield ModelEvent("done", response=response)
