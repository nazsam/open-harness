"""OpenAI Chat Completions adapter.

Also used for any OpenAI-compatible server: Ollama, vLLM, LM Studio, Groq,
Together, OpenRouter, DeepSeek, Mistral and others. Point ``base_url`` at the
server.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from ..errors import ConfigError, ModelError
from ..schema import is_strict_compatible
from ..types import Message, ModelRequest, ModelResponse, ToolCall, Usage, new_id
from .base import HTTPClient, Model, ModelEvent, parse_args

OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIModel(Model):
    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        provider_name: str = "openai",
        headers: dict[str, str] | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        require_key: bool = True,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.provider = provider_name
        key = api_key or os.environ.get(api_key_env)
        if not key and require_key:
            raise ConfigError(f"{provider_name}: set {api_key_env} or pass api_key=")
        if base_url is None and provider_name == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL")
        self.base_url = base_url or OPENAI_BASE_URL
        self.official = self.base_url.rstrip("/") == OPENAI_BASE_URL
        h = {"Content-Type": "application/json", **(headers or {})}
        if key:
            h["Authorization"] = f"Bearer {key}"
        self.http = HTTPClient(self.base_url, h, timeout=timeout, max_retries=max_retries, provider=provider_name)

    async def aclose(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------- request

    def build_body(self, req: ModelRequest, stream: bool) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            messages.extend(self._convert(m))
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if req.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in req.tools
            ]
        s = req.settings
        if s.tool_choice and req.tools:
            body["tool_choice"] = (
                s.tool_choice
                if s.tool_choice in ("auto", "required", "none")
                else {"type": "function", "function": {"name": s.tool_choice}}
            )
        if s.parallel_tool_calls is not None and req.tools:
            body["parallel_tool_calls"] = s.parallel_tool_calls
        if s.temperature is not None:
            body["temperature"] = s.temperature
        if s.top_p is not None:
            body["top_p"] = s.top_p
        if s.max_output_tokens is not None:
            body["max_completion_tokens" if self.official else "max_tokens"] = s.max_output_tokens
        if req.output_schema:
            strict = req.output_schema.strict and is_strict_compatible(req.output_schema.schema)
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": req.output_schema.name, "schema": req.output_schema.schema, "strict": strict},
            }
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        body.update(s.extra)
        return body

    def _convert(self, m: Message) -> list[dict[str, Any]]:
        if m.role == "system":
            return [{"role": "system", "content": m.content}]
        if m.role == "user":
            return [{"role": "user", "content": m.content}]
        if m.role == "tool":
            return [{"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or "(empty)"}]
        out: dict[str, Any] = {"role": "assistant", "content": m.content or None}
        if m.tool_calls:
            out["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in m.tool_calls
            ]
        return [out]

    # ------------------------------------------------------------- response

    @staticmethod
    def _usage(u: dict[str, Any] | None) -> Usage:
        u = u or {}
        return Usage(
            input_tokens=u.get("prompt_tokens", 0) or 0, output_tokens=u.get("completion_tokens", 0) or 0, requests=1
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        data = await self.http.post_json("chat/completions", self.build_body(request, stream=False))
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"{self.provider}: response had no choices", body=data, provider=self.provider)
        msg = choices[0].get("message") or {}
        if msg.get("refusal"):
            text = msg["refusal"]
        else:
            text = msg.get("content") or ""
        calls = [
            ToolCall(
                id=tc.get("id") or new_id("call"),
                name=tc["function"]["name"],
                arguments=parse_args(tc["function"].get("arguments")),
            )
            for tc in msg.get("tool_calls") or []
        ]
        return ModelResponse(
            Message.assistant(text, calls), self._usage(data.get("usage")), choices[0].get("finish_reason")
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        text: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage(requests=1)
        finish = None
        async for _event, data in self.http.post_sse("chat/completions", self.build_body(request, stream=True)):
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise ModelError(f"{self.provider}: {chunk['error']}", body=chunk, provider=self.provider)
            if chunk.get("usage"):
                usage = self._usage(chunk["usage"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                    yield ModelEvent("text_delta", {"delta": delta["content"]})
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        tool_calls = []
        for idx in sorted(calls):
            c = calls[idx]
            tc = ToolCall(id=c["id"] or new_id("call"), name=c["name"], arguments=parse_args(c["args"]))
            tool_calls.append(tc)
            yield ModelEvent("tool_call", tc.to_dict())
        yield ModelEvent("done", response=ModelResponse(Message.assistant("".join(text), tool_calls), usage, finish))


# OpenAI-compatible presets: provider name -> (base_url, api key env var, key required)
COMPATIBLE_PRESETS: dict[str, tuple[str, str, bool]] = {
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", False),
    "lmstudio": ("http://localhost:1234/v1", "LMSTUDIO_API_KEY", False),
    "vllm": ("http://localhost:8000/v1", "VLLM_API_KEY", False),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", True),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", True),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", True),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", True),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", True),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY", True),
}
