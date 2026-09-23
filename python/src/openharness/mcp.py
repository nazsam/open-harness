"""Model Context Protocol (MCP) client.

Connect an agent to any MCP server and its tools become agent tools:

    fs = MCPServerStdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
    remote = MCPServerHTTP("https://example.com/mcp", headers={"Authorization": "Bearer ..."})
    agent = Agent(..., mcp_servers=[fs, remote])

The client targets MCP revision 2026-07-28 (stateless, per-request metadata)
and falls back to the legacy ``initialize`` handshake (2025-11-25 and earlier)
when a server does not speak the modern protocol, as the spec's dual-era
rules describe.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx

from ._version import __version__
from .errors import MCPError, ToolError
from .tools import Tool

MODERN_VERSION = "2026-07-28"
LEGACY_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26"]
CLIENT_INFO = {"name": "openharness", "version": __version__}
MODERN_ERROR_CODES = {-32020, -32021, -32022}


def _is_modern_error(body: Any) -> bool:
    err = body.get("error") if isinstance(body, dict) else None
    return isinstance(err, dict) and err.get("code") in MODERN_ERROR_CODES


def _safe_header(value: str) -> str:
    """Encode a header value using the spec's Base64 sentinel when needed."""
    plain = all(0x20 <= ord(c) <= 0x7E or c == "\t" for c in value) and value == value.strip()
    sentinel = value.startswith("=?base64?") and value.endswith("?=")
    if plain and not sentinel:
        return value
    return "=?base64?" + base64.b64encode(value.encode("utf-8")).decode("ascii") + "?="


_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def _header_params(schema: dict[str, Any]) -> list[tuple[list[str], str]] | None:
    """Collect ``x-mcp-header`` annotations. Returns None when the tool definition is invalid."""
    found: list[tuple[list[str], str]] = []
    seen: set[str] = set()

    def walk(node: dict[str, Any], path: list[str]) -> bool:
        for key, sub in (node.get("properties") or {}).items():
            if not isinstance(sub, dict):
                continue
            h = sub.get("x-mcp-header")
            if h is not None:
                t = sub.get("type")
                if (
                    not isinstance(h, str)
                    or not h
                    or not _TOKEN.match(h)
                    or h.lower() in seen
                    or t not in ("string", "integer", "boolean")
                ):
                    return False
                seen.add(h.lower())
                found.append(([*path, key], h))
            if sub.get("type") == "object" and not walk(sub, [*path, key]):
                return False
        return True

    return found if walk(schema, []) else None


def _content_to_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for c in result.get("content") or []:
        t = c.get("type")
        if t == "text":
            parts.append(c.get("text", ""))
        elif t == "image":
            parts.append(f"[image {c.get('mimeType', '')}]")
        elif t == "audio":
            parts.append(f"[audio {c.get('mimeType', '')}]")
        elif t == "resource":
            r = c.get("resource") or {}
            parts.append(r.get("text") or f"[resource {r.get('uri', '')}]")
        elif t == "resource_link":
            parts.append(f"[resource {c.get('uri', '')}]")
    if not parts and result.get("structuredContent") is not None:
        return json.dumps(result["structuredContent"], ensure_ascii=False)
    return "\n".join(parts)


class MCPServer(ABC):
    """Base class for MCP connections. Use ``MCPServerStdio`` or ``MCPServerHTTP``."""

    def __init__(
        self,
        name: str,
        *,
        cache_tools: bool = True,
        tool_filter: Callable[[str], bool] | None = None,
        tool_prefix: str = "",
        timeout: float = 60.0,
        needs_approval: bool = False,
    ):
        self.name = name
        self.cache_tools = cache_tools
        self.tool_filter = tool_filter
        self.tool_prefix = tool_prefix
        self.timeout = timeout
        self.needs_approval = needs_approval
        self.era: str | None = None  # "modern" | "legacy"
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self._tools: list[Tool] | None = None
        self._tool_defs: dict[str, dict[str, Any]] = {}
        self._next_id = 0
        self._connect_lock = asyncio.Lock()

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _meta(self) -> dict[str, Any]:
        return {
            "io.modelcontextprotocol/protocolVersion": self.protocol_version or MODERN_VERSION,
            "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            "io.modelcontextprotocol/clientCapabilities": {},
        }

    @abstractmethod
    async def _detect(self) -> None: ...

    @abstractmethod
    async def _send(self, method: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def connect(self) -> None:
        async with self._connect_lock:
            if self.era is None:
                await self._detect()

    async def request(
        self, method: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        await self.connect()
        params = dict(params or {})
        if self.era == "modern":
            params["_meta"] = {**params.get("_meta", {}), **self._meta()}
        result = await self._send(method, params, headers)
        if isinstance(result, dict) and result.get("resultType", "complete") == "input_required":
            raise MCPError(
                f"MCP server '{self.name}' asked for extra client input (elicitation/sampling), "
                "which this client does not provide."
            )
        return result

    async def list_tools(self) -> list[Tool]:
        if self._tools is not None and self.cache_tools:
            return self._tools
        defs: list[dict[str, Any]] = []
        cursor = None
        while True:
            res = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            defs.extend(res.get("tools") or [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
        tools = []
        self._tool_defs = {}
        for d in defs:
            if self.tool_filter and not self.tool_filter(d["name"]):
                continue
            if isinstance(self, MCPServerHTTP) and _header_params(d.get("inputSchema") or {}) is None:
                continue  # spec: exclude tools with invalid x-mcp-header annotations
            self._tool_defs[d["name"]] = d
            tools.append(self._wrap(d))
        self._tools = tools
        return tools

    def invalidate_tools_cache(self) -> None:
        self._tools = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.request(
            "tools/call", {"name": name, "arguments": arguments}, headers=self._call_headers(name, arguments)
        )

    def _call_headers(self, name: str, arguments: dict[str, Any]) -> dict[str, str]:
        return {}

    def _wrap(self, d: dict[str, Any]) -> Tool:
        remote_name = d["name"]
        schema = d.get("inputSchema") or {"type": "object", "properties": {}}

        async def handler(args: dict[str, Any], _ctx: Any) -> Any:
            res = await asyncio.wait_for(self.call_tool(remote_name, args), self.timeout)
            text = _content_to_text(res)
            if res.get("isError"):
                raise ToolError(text or "MCP tool reported an error")
            return text

        return Tool(
            name=f"{self.tool_prefix}{remote_name}",
            description=d.get("description") or d.get("title") or "",
            parameters=schema,
            handler=handler,
            needs_approval=self.needs_approval,
            metadata={"mcp_server": self.name, "annotations": d.get("annotations")},
        )

    async def __aenter__(self) -> MCPServer:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# ----------------------------------------------------------------------- stdio


class MCPServerStdio(MCPServer):
    """Launch an MCP server as a subprocess and talk over stdin/stdout."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        name: str | None = None,
        probe_timeout: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(name or os.path.basename(command), **kwargs)
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self.probe_timeout = probe_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self.stderr_tail: list[str] = []

    async def _start(self) -> None:
        if self._proc and self._proc.returncode is None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(self.env or {})},
            cwd=self.cwd,
            limit=16 * 1024 * 1024,
        )
        self._reader = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        async for line in self._proc.stderr:
            self.stderr_tail = (self.stderr_tail + [line.decode("utf-8", "replace").rstrip()])[-20:]

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            async for raw in self._proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in msg and ("result" in msg or "error" in msg):
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                elif "id" in msg and "method" in msg:  # legacy servers may send requests (ping, roots/list)
                    reply: dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"]}
                    if msg["method"] == "ping":
                        reply["result"] = {}
                    else:
                        reply["error"] = {"code": -32601, "message": "Method not supported by client"}
                    await self._write(reply)
        finally:
            err = MCPError(f"MCP server '{self.name}' exited. stderr: {' | '.join(self.stderr_tail[-5:])}")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()

    async def _write(self, msg: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _rpc(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        await self._start()
        rid = self._id()
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        try:
            return await asyncio.wait_for(fut, timeout or self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            await self._write(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": rid, "reason": "timeout"},
                }
            )
            raise

    async def _detect(self) -> None:
        await self._start()
        self.protocol_version = MODERN_VERSION
        try:
            msg = await self._rpc("server/discover", {"_meta": self._meta()}, timeout=self.probe_timeout)
        except asyncio.TimeoutError:
            msg = None
        if msg and "result" in msg:
            res = msg["result"]
            supported = res.get("supportedVersions") or [MODERN_VERSION]
            self.era = "modern"
            self.protocol_version = MODERN_VERSION if MODERN_VERSION in supported else supported[0]
            self.server_info = res.get("serverInfo") or {}
            return
        if msg and _is_modern_error(msg):
            supported = (msg["error"].get("data") or {}).get("supported") or []
            modern = [v for v in supported if v >= MODERN_VERSION]
            if modern:
                self.era, self.protocol_version = "modern", modern[0]
                return
            raise MCPError(f"MCP server '{self.name}' supports no compatible version: {supported}")
        await self._legacy_init()

    async def _legacy_init(self) -> None:
        msg = await self._rpc(
            "initialize", {"protocolVersion": LEGACY_VERSIONS[0], "capabilities": {}, "clientInfo": CLIENT_INFO}
        )
        if "error" in msg:
            e = msg["error"]
            raise MCPError(f"MCP initialize failed: {e.get('message')}", e.get("code"), e.get("data"))
        res = msg["result"]
        self.era = "legacy"
        self.protocol_version = res.get("protocolVersion", LEGACY_VERSIONS[0])
        self.server_info = res.get("serverInfo") or {}
        await self._write({"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def _send(self, method: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        msg = await self._rpc(method, params)
        if "error" in msg:
            e = msg["error"]
            raise MCPError(f"MCP {method} failed: {e.get('message')}", e.get("code"), e.get("data"))
        return msg.get("result") or {}

    async def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.stdin and not proc.stdin.is_closing():
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), 3)
        except asyncio.TimeoutError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 2)
            except asyncio.TimeoutError:
                proc.kill()
        for task in (self._reader, self._stderr_task):
            if task:
                task.cancel()
        self.era = None
        self._tools = None


# ------------------------------------------------------------------------ HTTP


class MCPServerHTTP(MCPServer):
    """Connect to a remote MCP server over Streamable HTTP."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        name: str | None = None,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(name or httpx.URL(url).host or "mcp", **kwargs)
        self.url = url
        self.headers = headers or {}
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=15.0))
        self._session_id: str | None = None

    def _call_headers(self, name: str, arguments: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.era != "modern":
            return out
        schema = (self._tool_defs.get(name) or {}).get("inputSchema") or {}
        for path, header in _header_params(schema) or []:
            value: Any = arguments
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                continue
            text = ("true" if value else "false") if isinstance(value, bool) else str(value)
            out[f"Mcp-Param-{header}"] = _safe_header(text)
        return out

    async def _post(self, body: dict[str, Any], extra: dict[str, str]) -> tuple[int, Any, httpx.Headers]:
        if self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=15.0))
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
            **extra,
        }
        async with self._client.stream("POST", self.url, json=body, headers=headers) as resp:
            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                from .models.base import iter_sse

                async for _ev, data in iter_sse(resp.aiter_lines()):
                    try:
                        msg = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("id") == body.get("id") and ("result" in msg or "error" in msg):
                        return resp.status_code, msg, resp.headers
                return resp.status_code, None, resp.headers
            raw = await resp.aread()
            try:
                return resp.status_code, json.loads(raw) if raw else None, resp.headers
            except json.JSONDecodeError:
                return resp.status_code, raw.decode("utf-8", "replace"), resp.headers

    def _modern_headers(self, method: str, params: dict[str, Any]) -> dict[str, str]:
        h = {"MCP-Protocol-Version": self.protocol_version or MODERN_VERSION, "Mcp-Method": method}
        name = (
            params.get("name")
            if method in ("tools/call", "prompts/get")
            else params.get("uri")
            if method == "resources/read"
            else None
        )
        if name:
            h["Mcp-Name"] = _safe_header(str(name))
        return h

    async def _detect(self) -> None:
        self.protocol_version = MODERN_VERSION
        params = {"_meta": self._meta()}
        body = {"jsonrpc": "2.0", "id": self._id(), "method": "server/discover", "params": params}
        try:
            status, msg, _ = await self._post(body, self._modern_headers("server/discover", params))
        except httpx.HTTPError as e:
            raise MCPError(f"Cannot reach MCP server at {self.url}: {e}") from e
        if status < 400 and isinstance(msg, dict) and "result" in msg:
            res = msg["result"]
            self.era = "modern"
            supported = res.get("supportedVersions") or [MODERN_VERSION]
            self.protocol_version = MODERN_VERSION if MODERN_VERSION in supported else supported[0]
            self.server_info = res.get("serverInfo") or {}
            return
        if isinstance(msg, dict) and _is_modern_error(msg):
            supported = (msg["error"].get("data") or {}).get("supported") or []
            modern = [v for v in supported if v >= MODERN_VERSION]
            if not modern:
                raise MCPError(f"MCP server '{self.name}' supports no compatible version: {supported}")
            self.era, self.protocol_version = "modern", modern[0]
            return
        if status == 404 and isinstance(msg, dict) and (msg.get("error") or {}).get("code") == -32601:
            self.era = "modern"  # modern server without discover; per-request negotiation still works
            return
        await self._legacy_init()

    async def _legacy_init(self) -> None:
        body = {
            "jsonrpc": "2.0",
            "id": self._id(),
            "method": "initialize",
            "params": {"protocolVersion": LEGACY_VERSIONS[0], "capabilities": {}, "clientInfo": CLIENT_INFO},
        }
        status, msg, headers = await self._post(body, {})
        if status >= 400 or not isinstance(msg, dict) or "result" not in msg:
            raise MCPError(
                f"MCP server at {self.url} rejected both modern and legacy connections (HTTP {status}): {msg}"
            )
        self.era = "legacy"
        self.protocol_version = msg["result"].get("protocolVersion", LEGACY_VERSIONS[0])
        self.server_info = msg["result"].get("serverInfo") or {}
        self._session_id = headers.get("mcp-session-id")
        await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, self._legacy_headers())

    def _legacy_headers(self) -> dict[str, str]:
        h = {"MCP-Protocol-Version": self.protocol_version or LEGACY_VERSIONS[0]}
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    async def _send(self, method: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        body = {"jsonrpc": "2.0", "id": self._id(), "method": method, "params": params}
        extra = (
            {**self._modern_headers(method, params), **(headers or {})}
            if self.era == "modern"
            else self._legacy_headers()
        )
        status, msg, _ = await self._post(body, extra)
        if not isinstance(msg, dict):
            raise MCPError(f"MCP {method}: unexpected HTTP {status} response: {msg!r}"[:500])
        if "error" in msg:
            e = msg["error"]
            raise MCPError(f"MCP {method} failed: {e.get('message')}", e.get("code"), e.get("data"))
        return msg.get("result") or {}

    async def close(self) -> None:
        if self.era == "legacy" and self._session_id:
            try:
                await self._client.delete(self.url, headers={**self.headers, **self._legacy_headers()})
            except httpx.HTTPError:
                pass
        await self._client.aclose()
        self.era = None
        self._tools = None
