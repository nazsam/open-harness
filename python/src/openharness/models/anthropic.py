"""Anthropic Messages API adapter (Claude models)."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from ..errors import ConfigError, ModelError
from ..types import Message, ModelRequest, ModelResponse, ToolCall, Usage
from .base import HTTPClient, Model, ModelEvent, parse_args

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 16000


class AnthropicModel(Model):
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.model = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError("anthropic: set ANTHROPIC_API_KEY or pass api_key=")
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or ANTHROPIC_BASE_URL
        self.http = HTTPClient(
            self.base_url,
            {
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
                **(headers or {}),
            },
            timeout=timeout,
            max_retries=max_retries,
            provider="anthropic",
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------- request

    def build_body(self, req: ModelRequest, stream: bool) -> dict[str, Any]:
        s = req.settings
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": s.max_output_tokens or DEFAULT_MAX_TOKENS,
            "messages": self._messages(req.messages),
        }
        system_parts = [req.system] if req.system else []
        system_parts += [m.content for m in req.messages if m.role == "system" and m.content]
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if req.tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in req.tools
            ]
            if s.tool_choice:
                tc = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}.get(
                    s.tool_choice, {"type": "tool", "name": s.tool_choice}
                )
                if s.parallel_tool_calls is False and tc["type"] != "none":
                    tc["disable_parallel_tool_use"] = True
                body["tool_choice"] = tc
            elif s.parallel_tool_calls is False:
                body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if s.temperature is not None:
            body["temperature"] = s.temperature
        if s.top_p is not None:
            body["top_p"] = s.top_p
        if req.output_schema:
            body["output_config"] = {"format": {"type": "json_schema", "schema": req.output_schema.schema}}
        if stream:
            body["stream"] = True
        body.update(s.extra)
        return body

    def _messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict[str, Any]]) -> None:
            if not blocks:
                return
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for m in messages:
            if m.role == "system":
                continue
            if m.role == "user":
                push("user", [{"type": "text", "text": m.content or "(empty)"}])
            elif m.role == "tool":
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content or "(empty)",
                }
                if m.is_error:
                    block["is_error"] = True
                push("user", [block])
            else:
                raw = self.own_raw(m)
                if raw:
                    push("assistant", [dict(b) for b in raw])
                    continue
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                push("assistant", blocks or [{"type": "text", "text": "(empty)"}])
        # The Messages API requires tool_result blocks to come first in a user turn.
        for msg in out:
            if msg["role"] == "user":
                msg["content"].sort(key=lambda b: 0 if b.get("type") == "tool_result" else 1)
        return out

    # ------------------------------------------------------------- response

    def _to_message(self, blocks: list[dict[str, Any]]) -> Message:
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [
            ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        msg = Message.assistant(text, calls)
        msg.raw = {"provider": self.provider, "data": blocks}
        return msg

    @staticmethod
    def _usage(u: dict[str, Any] | None) -> Usage:
        u = u or {}
        inp = (
            (u.get("input_tokens") or 0)
            + (u.get("cache_read_input_tokens") or 0)
            + (u.get("cache_creation_input_tokens") or 0)
        )
        return Usage(input_tokens=inp, output_tokens=u.get("output_tokens") or 0, requests=1)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        data = await self.http.post_json("messages", self.build_body(request, stream=False))
        if data.get("type") == "error":
            raise ModelError(f"anthropic: {data.get('error')}", body=data, provider=self.provider)
        return ModelResponse(
            self._to_message(data.get("content") or []), self._usage(data.get("usage")), data.get("stop_reason")
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        blocks: dict[int, dict[str, Any]] = {}
        partial_json: dict[int, str] = {}
        usage = Usage(requests=1)
        stop = None
        async for event, data in self.http.post_sse("messages", self.build_body(request, stream=True)):
            payload = json.loads(data)
            etype = payload.get("type") or event
            if etype == "message_start":
                usage = self._usage(payload.get("message", {}).get("usage"))
            elif etype == "content_block_start":
                idx = payload["index"]
                blocks[idx] = dict(payload["content_block"])
                if blocks[idx].get("type") == "tool_use":
                    partial_json[idx] = ""
            elif etype == "content_block_delta":
                idx, delta = payload["index"], payload["delta"]
                b = blocks.setdefault(idx, {"type": "text", "text": ""})
                dt = delta.get("type")
                if dt == "text_delta":
                    b["text"] = b.get("text", "") + delta["text"]
                    yield ModelEvent("text_delta", {"delta": delta["text"]})
                elif dt == "input_json_delta":
                    partial_json[idx] = partial_json.get(idx, "") + delta.get("partial_json", "")
                elif dt == "thinking_delta":
                    b["thinking"] = b.get("thinking", "") + delta.get("thinking", "")
                elif dt == "signature_delta":
                    b["signature"] = b.get("signature", "") + delta.get("signature", "")
                elif dt == "citations_delta":
                    b.setdefault("citations", []).append(delta.get("citation"))
            elif etype == "content_block_stop":
                idx = payload["index"]
                if idx in partial_json:
                    blocks[idx]["input"] = parse_args(partial_json.pop(idx))
                    b = blocks[idx]
                    yield ModelEvent("tool_call", {"id": b["id"], "name": b["name"], "arguments": b["input"]})
            elif etype == "message_delta":
                stop = (payload.get("delta") or {}).get("stop_reason") or stop
                out_tokens = (payload.get("usage") or {}).get("output_tokens")
                if out_tokens is not None:
                    usage.output_tokens = out_tokens
            elif etype == "error":
                raise ModelError(
                    f"anthropic stream error: {payload.get('error')}", body=payload, provider=self.provider
                )
        ordered = [blocks[i] for i in sorted(blocks)]
        yield ModelEvent("done", response=ModelResponse(self._to_message(ordered), usage, stop))
