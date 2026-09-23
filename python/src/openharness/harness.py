"""``Harness``: a working agent in one line, with sensible defaults wired up.

    from openharness import Harness

    h = Harness()                        # model from env, calculator + clock tools, in-memory session
    print(h.ask_sync("What is 17% of 2,340?"))

The harness owns an agent, a session, optional long-term memory, MCP
connections, limits and tracing. Everything is swappable, and ``h.agent``
is a normal ``Agent`` you can pass to ``run`` yourself.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .agent import Agent
from .builtin_tools import DEFAULT_TOOLS
from .context import CancelToken, RunLimits
from .errors import RunStopped
from .mcp import MCPServer
from .memory import FileMemory, InMemoryMemory, Memory
from .models import Model
from .result import RunResult
from .runner import Approver, run, run_stream
from .sessions import FileSession, InMemorySession, Session
from .tracing import TraceProcessor
from .types import ModelSettings, Usage

DEFAULT_INSTRUCTIONS = (
    "You are a capable, concise assistant. Use tools when they help you give a correct answer, "
    "and say so plainly when you are unsure."
)


def _session(value: Session | str | Path | None) -> Session:
    if value is None or value == "memory":
        return InMemorySession()
    if isinstance(value, (str, Path)):
        return FileSession(value)
    return value


def _memory(value: Memory | str | Path | bool | None) -> Memory | None:
    if value is None or value is False:
        return None
    if value is True:
        return InMemoryMemory()
    if isinstance(value, (str, Path)):
        return FileMemory(value)
    return value


class Harness:
    def __init__(
        self,
        model: str | Model | None = None,
        *,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        agent: Agent | None = None,
        session: Session | str | Path | None = None,
        memory: Memory | str | Path | bool | None = None,
        mcp_servers: list[MCPServer] | None = None,
        limits: RunLimits | None = None,
        model_settings: ModelSettings | None = None,
        trace: TraceProcessor | list[TraceProcessor] | None = None,
        approve: Approver | None = None,
        name: str = "assistant",
    ):
        self.agent = agent or Agent(
            name=name,
            instructions=instructions or DEFAULT_INSTRUCTIONS,
            model=model,
            tools=list(DEFAULT_TOOLS if tools is None else tools),
            mcp_servers=mcp_servers or [],
            memory=_memory(memory),
            model_settings=model_settings or ModelSettings(),
        )
        self.session = _session(session)
        self.limits = limits or RunLimits()
        self.trace = trace
        self.approve = approve
        self.usage = Usage()
        self._cancel: CancelToken | None = None

    def _kwargs(self, **extra: Any) -> dict[str, Any]:
        self._cancel = CancelToken()
        return {
            "session": self.session,
            "limits": self.limits,
            "trace": self.trace,
            "approve": self.approve,
            "cancel_token": self._cancel,
            **extra,
        }

    async def run(self, prompt: str, **kwargs: Any) -> RunResult:
        result = await run(self.agent, prompt, **self._kwargs(**kwargs))
        self.usage.add(result.usage)
        return result

    async def ask(self, prompt: str, **kwargs: Any) -> Any:
        """Send a message and return the answer (text, or structured output)."""
        return (await self.run(prompt, **kwargs)).output

    def ask_sync(self, prompt: str, **kwargs: Any) -> Any:
        return asyncio.run(self.ask(prompt, **kwargs))

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text as it is generated."""
        s = run_stream(self.agent, prompt, **self._kwargs(**kwargs))
        async for event in s:
            if event.type == "text_delta":
                yield event.data["delta"]
        self.usage.add((await s.result()).usage)

    def cancel(self) -> None:
        if self._cancel:
            self._cancel.cancel("cancelled by user")

    async def reset(self) -> None:
        await self.session.clear()

    async def aclose(self) -> None:
        for server in self.agent.mcp_servers:
            await server.close()

    async def __aenter__(self) -> Harness:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------- terminal chat

    async def chat(self, *, show_tools: bool = True, input_fn: Any = None, out: Any = None) -> None:
        """Interactive terminal chat. Commands: /exit /reset /usage /tools /help."""
        from .cli import terminal_chat

        await terminal_chat(self, show_tools=show_tools, input_fn=input_fn, out=out or sys.stdout)


__all__ = ["Harness", "RunStopped"]
