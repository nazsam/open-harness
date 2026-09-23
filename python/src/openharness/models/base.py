"""The ``Model`` interface every provider adapter implements, plus shared HTTP plumbing."""

from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import ModelError
from ..types import Message, ModelRequest, ModelResponse, Usage


@dataclass
class ModelEvent:
    """Streaming event from a provider.

    ``type`` is ``text_delta`` (data: delta), ``tool_call`` (data: id, name,
    arguments) or ``done`` (response: the complete ModelResponse).
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    response: ModelResponse | None = None


class Model(ABC):
    """A chat model. Implement ``generate``; ``stream`` has a non-streaming fallback."""

    provider: str = "custom"
    model: str = ""

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse: ...

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        response = await self.generate(request)
        if response.message.content:
            yield ModelEvent("text_delta", {"delta": response.message.content})
        for tc in response.message.tool_calls:
            yield ModelEvent("tool_call", tc.to_dict())
        yield ModelEvent("done", response=response)

    async def aclose(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.provider}:{self.model})"

    # helpers for adapters -------------------------------------------------

    def own_raw(self, message: Message) -> Any:
        """Provider payload saved on ``message`` if it came from this provider."""
        if message.raw and message.raw.get("provider") == self.provider:
            return message.raw.get("data")
        return None


RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class HTTPClient:
    """Small async HTTP wrapper with retries and Server-Sent Events parsing."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        provider: str = "http",
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.max_retries = max_retries
        self.provider = provider
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    def _url(self, path: str) -> str:
        return path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"

    async def _backoff(self, attempt: int, resp: httpx.Response | None) -> None:
        delay = min(2**attempt, 30) + random.random()
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                try:
                    delay = min(float(ra), 60.0)
                except ValueError:
                    pass
        await asyncio.sleep(delay)

    def _error(self, resp: httpx.Response, body: str) -> ModelError:
        try:
            data = json.loads(body)
        except ValueError:
            data = body
        msg = data
        if isinstance(data, dict):
            err = data.get("error", data)
            msg = err.get("message", err) if isinstance(err, dict) else err
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            msg = data[0].get("error", {}).get("message", data)
        return ModelError(
            f"{self.provider} API error {resp.status_code}: {msg}",
            status=resp.status_code,
            body=data,
            provider=self.provider,
        )

    async def post_json(self, path: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(self._url(path), json=body, headers={**self.headers, **(headers or {})})
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                if attempt >= self.max_retries:
                    raise ModelError(f"{self.provider}: network error: {e}", provider=self.provider) from e
                await self._backoff(attempt, None)
                continue
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                await self._backoff(attempt, resp)
                continue
            if resp.status_code >= 400:
                raise self._error(resp, resp.text)
            return resp.json()
        raise ModelError(f"{self.provider}: request failed", provider=self.provider)

    async def post_sse(
        self, path: str, body: dict[str, Any], headers: dict[str, str] | None = None
    ) -> AsyncIterator[tuple[str | None, str]]:
        """POST and yield (event, data) pairs from an SSE response."""
        for attempt in range(self.max_retries + 1):
            req = self._client.build_request(
                "POST",
                self._url(path),
                json=body,
                headers={**self.headers, "Accept": "text/event-stream", **(headers or {})},
            )
            try:
                resp = await self._client.send(req, stream=True)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                if attempt >= self.max_retries:
                    raise ModelError(f"{self.provider}: network error: {e}", provider=self.provider) from e
                await self._backoff(attempt, None)
                continue
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")
                await resp.aclose()
                if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                    await self._backoff(attempt, resp)
                    continue
                raise self._error(resp, text)
            try:
                async for item in iter_sse(resp.aiter_lines()):
                    yield item
            finally:
                await resp.aclose()
            return


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str | None, str]]:
    event: str | None = None
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if key == "event":
            event = value
        elif key == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


def parse_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse tool-call arguments that arrive as a JSON string."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}


def empty_usage() -> Usage:
    return Usage(requests=1)
