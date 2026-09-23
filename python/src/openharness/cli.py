"""Command line interface.

openharness chat --model anthropic:claude-sonnet-5
openharness chat --model ollama:llama3.2 --mcp "npx -y @modelcontextprotocol/server-filesystem ."
openharness run "What is 2**32?" --model openai:gpt-6-luna
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import sys
import threading
from typing import Any, TextIO

from ._version import __version__
from .builtin_tools import ALL_TOOLS
from .context import RunContext, RunLimits
from .errors import OpenHarnessError, RunCancelled, RunStopped
from .mcp import MCPServerHTTP, MCPServerStdio
from .models import DEFAULT_MODELS, PROVIDERS
from .tracing import ConsoleProcessor, JSONLProcessor
from .types import ToolCall

DIM, BOLD, CYAN, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[31m", "\033[0m"


def _color(out: TextIO) -> bool:
    return hasattr(out, "isatty") and out.isatty() and not os.environ.get("NO_COLOR")


async def _ainput(read: Any, prompt: str) -> str:
    """Read a line in a daemon thread so Ctrl+C at the prompt can still exit cleanly."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()

    def worker() -> None:
        try:
            value = read(prompt)
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(value))
        except BaseException as exc:  # EOFError, KeyboardInterrupt
            err = exc
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_exception(err))

    threading.Thread(target=worker, daemon=True).start()
    return await fut


def _approver(auto_yes: bool) -> Any:
    async def approve(_ctx: RunContext, call: ToolCall) -> bool:
        if auto_yes:
            return True
        args = json.dumps(call.arguments)[:300]
        answer = await _ainput(input, f"\nAllow tool {call.name}({args})? [y/N] ")
        return answer.strip().lower() in ("y", "yes")

    return approve


async def terminal_chat(
    harness: Any, *, show_tools: bool = True, input_fn: Any = None, out: TextIO = sys.stdout
) -> None:
    from .runner import run_stream

    c = _color(out)
    dim, bold, cyan, red, reset = (DIM, BOLD, CYAN, RED, RESET) if c else ("",) * 5
    read = input_fn or (lambda prompt: input(prompt))
    agent = harness.agent
    try:
        model = agent.get_model()
        label = f"{model.provider}:{model.model}"
    except OpenHarnessError as e:
        print(f"{red}{e}{reset}", file=out)
        return
    print(
        f"{bold}openharness {__version__}{reset}  model {cyan}{label}{reset}  "
        f"tools {len(agent.tools)}  {dim}(/help for commands, Ctrl+C stops a reply){reset}",
        file=out,
    )
    while True:
        try:
            line = await _ainput(read, f"\n{bold}you>{reset} ")
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            break
        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            cmd = text.split()[0].lower()
            if cmd in ("/exit", "/quit", "/q"):
                break
            if cmd == "/reset":
                await harness.reset()
                print(f"{dim}conversation cleared{reset}", file=out)
            elif cmd == "/usage":
                print(f"{dim}{harness.usage.to_dict()}{reset}", file=out)
            elif cmd == "/tools":
                names = [t.name for t in agent.tools]
                for server in agent.mcp_servers:
                    names += [f"{t.name} (mcp:{server.name})" for t in await server.list_tools()]
                print(f"{dim}{', '.join(names) or 'no tools'}{reset}", file=out)
            else:
                print(
                    f"{dim}/reset  clear the conversation\n/usage  token usage\n/tools  list tools\n"
                    f"/exit   quit{reset}",
                    file=out,
                )
            continue
        stream = run_stream(agent, text, **harness._kwargs())
        print(f"{bold}{agent.name}>{reset} ", end="", file=out, flush=True)
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, stream.cancel, "interrupted")
            sig_installed = True
        except (NotImplementedError, RuntimeError, ValueError):  # Windows or not main thread
            sig_installed = False
        try:
            async for ev in stream:
                if ev.type == "text_delta":
                    print(ev.data["delta"], end="", file=out, flush=True)
                elif ev.type == "tool_call" and show_tools:
                    args = json.dumps(ev.data["arguments"], ensure_ascii=False)
                    print(f"\n{dim}  -> {ev.data['name']}({args[:200]}){reset}", file=out, flush=True)
                elif ev.type == "tool_result" and show_tools:
                    mark = "error" if ev.data["is_error"] else "ok"
                    preview = ev.data["output"].replace("\n", " ")[:160]
                    print(f"{dim}  <- {mark}: {preview}{reset}\n", end="", file=out, flush=True)
                elif ev.type == "handoff":
                    print(f"\n{dim}  handoff: {ev.data['from']} -> {ev.data['to']}{reset}\n", end="", file=out)
            result = await stream.result()
            harness.usage.add(result.usage)
            if not isinstance(result.output, str):
                print(json.dumps(result.output, default=str, indent=2), file=out)
            print(file=out)
        except (KeyboardInterrupt, RunCancelled):
            stream.cancel("interrupted")
            print(f"\n{dim}(stopped){reset}", file=out)
        except RunStopped as e:
            print(f"\n{red}stopped: {e}{reset}", file=out)
        except OpenHarnessError as e:
            print(f"\n{red}error: {e}{reset}", file=out)
        finally:
            if sig_installed:
                loop.remove_signal_handler(signal.SIGINT)


def _build_harness(args: argparse.Namespace) -> Any:
    from .harness import Harness

    if args.tools == "none":
        tools: list[Any] = []
    else:
        names = [n.strip() for n in args.tools.split(",") if n.strip()]
        unknown = [n for n in names if n not in ALL_TOOLS]
        if unknown:
            raise SystemExit(f"Unknown tools: {', '.join(unknown)}. Available: {', '.join(ALL_TOOLS)}")
        tools = [ALL_TOOLS[n] for n in names]
    servers: list[Any] = []
    for spec in args.mcp or []:
        parts = shlex.split(spec)
        servers.append(MCPServerStdio(parts[0], parts[1:]))
    for url in args.mcp_http or []:
        servers.append(MCPServerHTTP(url))
    trace = None
    if args.trace:
        trace = ConsoleProcessor() if args.trace in ("console", "1") else JSONLProcessor(args.trace)
    return Harness(
        args.model,
        instructions=args.system,
        tools=tools,
        session=args.session,
        memory=args.memory,
        mcp_servers=servers,
        limits=RunLimits(max_turns=args.max_turns, max_tool_calls=args.max_tool_calls, timeout_seconds=args.timeout),
        trace=trace,
        approve=_approver(args.yes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openharness", description="Chat with an AI agent from your terminal.")
    parser.add_argument("--version", action="version", version=f"openharness {__version__}")
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-m", "--model", help="provider:model, e.g. anthropic:claude-sonnet-5 or ollama:llama3.2")
        p.add_argument("-s", "--system", help="system instructions")
        p.add_argument("--session", help="save the conversation to this .jsonl file")
        p.add_argument("--memory", help="long-term memory JSON file (enables remember/recall)")
        p.add_argument(
            "--tools", default="calculator,current_time", help=f"comma list from: {', '.join(ALL_TOOLS)}; or 'none'"
        )
        p.add_argument("--mcp", action="append", metavar="CMD", help="MCP stdio server command (repeatable)")
        p.add_argument("--mcp-http", action="append", metavar="URL", help="MCP HTTP server URL (repeatable)")
        p.add_argument("--max-turns", type=int, default=20)
        p.add_argument("--max-tool-calls", type=int, default=50)
        p.add_argument("--timeout", type=float, default=600, help="seconds per reply")
        p.add_argument("--trace", help="'console' or a .jsonl path")
        p.add_argument("-y", "--yes", action="store_true", help="auto-approve tools that need approval")

    common(sub.add_parser("chat", help="interactive chat (default)"))
    p_run = sub.add_parser("run", help="answer one prompt and exit")
    p_run.add_argument("prompt", nargs="+")
    common(p_run)
    sub.add_parser("models", help="list providers and default models")

    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["chat", *(argv or sys.argv[1:])])

    if args.command == "models":
        for p in PROVIDERS:
            print(f"{p:<11} {DEFAULT_MODELS.get(p, '(pass a model name)')}")
        return 0
    harness = _build_harness(args)

    async def go() -> int:
        try:
            if args.command == "run":
                prompt = " ".join(args.prompt)
                try:
                    piped = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
                except (OSError, ValueError):
                    piped = ""
                if piped.strip():  # e.g. `cat notes.txt | openharness run "summarize"`
                    prompt = f"{prompt}\n\n{piped}"
                async for chunk in harness.stream(prompt):
                    print(chunk, end="", flush=True)
                print()
            else:
                await harness.chat()
            return 0
        except RunStopped as e:
            print(f"stopped: {e}", file=sys.stderr)
            return 2
        except OpenHarnessError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        finally:
            await harness.aclose()

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
