"""Core data types shared by every part of the SDK.

Messages are stored in a provider-neutral format so a conversation can be
continued on a different model provider. Provider adapters may attach their
original payload in ``Message.raw`` so provider-specific details (for example
reasoning signatures) survive a round trip on the same provider.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(id=d["id"], name=d["name"], arguments=d.get("arguments") or {})


@dataclass
class Message:
    """One entry in a conversation.

    * ``user`` / ``system``: ``content`` holds the text.
    * ``assistant``: ``content`` holds text (may be empty) and ``tool_calls``
      any tool invocations requested by the model.
    * ``tool``: the result of one tool call, linked by ``tool_call_id``.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name for tool results, agent name for assistant
    is_error: bool = False
    raw: dict[str, Any] | None = None  # {"provider": str, "data": Any}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        if self.is_error:
            d["is_error"] = True
        if self.raw is not None:
            d["raw"] = self.raw
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            role=d["role"],
            content=d.get("content") or "",
            tool_calls=[ToolCall.from_dict(t) for t in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            is_error=bool(d.get("is_error", False)),
            raw=d.get("raw"),
        )

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=text)

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", content=text)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=text, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, name: str, content: str, is_error: bool = False) -> Message:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name, is_error=is_error)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.requests += other.requests

    def to_dict(self) -> dict[str, int]:
        return {**asdict(self), "total_tokens": self.total_tokens}


@dataclass
class ToolSpec:
    """What the model sees for a tool."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class OutputSchema:
    name: str
    schema: dict[str, Any]
    strict: bool = True


@dataclass
class ModelSettings:
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    tool_choice: str | None = None  # "auto" | "required" | "none" | <tool name>
    parallel_tool_calls: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # merged into the provider request body

    def merged(self, other: ModelSettings | None) -> ModelSettings:
        if other is None:
            return self
        out = ModelSettings(**{k: v for k, v in asdict(self).items()})
        for k, v in asdict(other).items():
            if k == "extra":
                out.extra = {**self.extra, **v}
            elif v is not None:
                setattr(out, k, v)
        return out


@dataclass
class ModelRequest:
    system: str | None
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    output_schema: OutputSchema | None = None
    settings: ModelSettings = field(default_factory=ModelSettings)


@dataclass
class ModelResponse:
    message: Message
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


# ----------------------------------------------------------------- stream events


@dataclass
class StreamEvent:
    """Events yielded while a run streams.

    ``type`` is one of:

    * ``agent_start``   data: {"agent"}
    * ``text_delta``    data: {"delta"}
    * ``tool_call``     data: {"id", "name", "arguments"}
    * ``tool_result``   data: {"id", "name", "output", "is_error"}
    * ``handoff``       data: {"from", "to"}
    * ``message``       data: {"message"}  (complete assistant message for a turn)
    * ``run_end``       data: {"result"}
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    agent: str | None = None
    timestamp: float = field(default_factory=time.time)


def to_json(value: Any) -> str:
    """Serialise a tool result for the model."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=_json_default, ensure_ascii=False)
    except TypeError:
        return str(value)


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    return str(o)
