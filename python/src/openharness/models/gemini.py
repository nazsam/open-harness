"""Google Gemini adapter (Gemini API ``generateContent``)."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from ..errors import ConfigError, ModelError
from ..types import Message, ModelRequest, ModelResponse, ToolCall, Usage, new_id
from .base import HTTPClient, Model, ModelEvent

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiModel(Model):
    provider = "gemini"

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
        self.model = model.removeprefix("models/")
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ConfigError("gemini: set GEMINI_API_KEY or pass api_key=")
        self.base_url = base_url or os.environ.get("GEMINI_BASE_URL") or GEMINI_BASE_URL
        self.http = HTTPClient(
            self.base_url,
            {"x-goog-api-key": key, "Content-Type": "application/json", **(headers or {})},
            timeout=timeout,
            max_retries=max_retries,
            provider="gemini",
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------- request

    def build_body(self, req: ModelRequest) -> dict[str, Any]:
        s = req.settings
        body: dict[str, Any] = {"contents": self._contents(req.messages)}
        system_parts = [req.system] if req.system else []
        system_parts += [m.content for m in req.messages if m.role == "system" and m.content]
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if req.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": t.name, "description": t.description, "parametersJsonSchema": t.parameters}
                        for t in req.tools
                    ]
                }
            ]
            if s.tool_choice:
                mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(s.tool_choice)
                cfg: dict[str, Any] = {"mode": mode or "ANY"}
                if mode is None:
                    cfg["allowedFunctionNames"] = [s.tool_choice]
                body["toolConfig"] = {"functionCallingConfig": cfg}
        gen: dict[str, Any] = {}
        if s.temperature is not None:
            gen["temperature"] = s.temperature
        if s.top_p is not None:
            gen["topP"] = s.top_p
        if s.max_output_tokens is not None:
            gen["maxOutputTokens"] = s.max_output_tokens
        if req.output_schema:
            gen["responseMimeType"] = "application/json"
            gen["responseJsonSchema"] = req.output_schema.schema
        if gen:
            body["generationConfig"] = gen
        body.update(s.extra)
        return body

    def _contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def push(role: str, parts: list[dict[str, Any]]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["parts"].extend(parts)
            else:
                out.append({"role": role, "parts": list(parts)})

        for m in messages:
            if m.role == "system":
                continue
            if m.role == "user":
                push("user", [{"text": m.content or " "}])
            elif m.role == "tool":
                try:
                    result: Any = json.loads(m.content)
                except (ValueError, TypeError):
                    result = m.content
                response = {"error": result} if m.is_error else {"result": result}
                fr: dict[str, Any] = {"name": m.name or "tool", "response": response}
                if m.tool_call_id and not m.tool_call_id.startswith("gcall_"):
                    fr["id"] = m.tool_call_id
                push("user", [{"functionResponse": fr}])
            else:
                raw = self.own_raw(m)
                if raw:
                    push("model", [dict(p) for p in raw])
                    continue
                parts: list[dict[str, Any]] = []
                if m.content:
                    parts.append({"text": m.content})
                for tc in m.tool_calls:
                    fc: dict[str, Any] = {"name": tc.name, "args": tc.arguments}
                    if not tc.id.startswith("gcall_"):
                        fc["id"] = tc.id
                    parts.append({"functionCall": fc})
                push("model", parts or [{"text": " "}])
        return out

    # ------------------------------------------------------------- response

    @staticmethod
    def _usage(u: dict[str, Any] | None) -> Usage:
        u = u or {}
        out = (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
        return Usage(input_tokens=u.get("promptTokenCount") or 0, output_tokens=out, requests=1)

    def _to_message(self, parts: list[dict[str, Any]]) -> Message:
        text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
        calls = []
        for p in parts:
            fc = p.get("functionCall")
            if fc:
                calls.append(
                    ToolCall(id=fc.get("id") or new_id("gcall"), name=fc["name"], arguments=fc.get("args") or {})
                )
        msg = Message.assistant(text, calls)
        msg.raw = {"provider": self.provider, "data": parts}
        return msg

    def _check(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        cands = data.get("candidates") or []
        if not cands:
            fb = data.get("promptFeedback") or {}
            if fb.get("blockReason"):
                raise ModelError(f"gemini: prompt blocked ({fb['blockReason']})", body=data, provider=self.provider)
            return []
        return cands

    async def generate(self, request: ModelRequest) -> ModelResponse:
        data = await self.http.post_json(f"models/{self.model}:generateContent", self.build_body(request))
        cands = self._check(data)
        parts = (cands[0].get("content") or {}).get("parts") or [] if cands else []
        finish = cands[0].get("finishReason") if cands else None
        return ModelResponse(self._to_message(parts), self._usage(data.get("usageMetadata")), finish)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        parts: list[dict[str, Any]] = []
        usage = Usage(requests=1)
        finish = None
        async for _event, data in self.http.post_sse(
            f"models/{self.model}:streamGenerateContent?alt=sse", self.build_body(request)
        ):
            chunk = json.loads(data)
            if chunk.get("error"):
                raise ModelError(f"gemini: {chunk['error']}", body=chunk, provider=self.provider)
            if chunk.get("usageMetadata"):
                usage = self._usage(chunk["usageMetadata"])
            for cand in self._check(chunk)[:1]:
                finish = cand.get("finishReason") or finish
                for p in (cand.get("content") or {}).get("parts") or []:
                    if (
                        "text" in p
                        and not p.get("thought")
                        and not p.get("thoughtSignature")
                        and parts
                        and "text" in parts[-1]
                        and not parts[-1].get("thought")
                        and not parts[-1].get("thoughtSignature")
                    ):
                        parts[-1]["text"] += p["text"]  # merge plain text chunks
                    else:
                        parts.append(dict(p))
                    if "text" in p and not p.get("thought") and p["text"]:
                        yield ModelEvent("text_delta", {"delta": p["text"]})
        msg = self._to_message(parts)
        for tc in msg.tool_calls:
            yield ModelEvent("tool_call", tc.to_dict())
        yield ModelEvent("done", response=ModelResponse(msg, usage, finish))
