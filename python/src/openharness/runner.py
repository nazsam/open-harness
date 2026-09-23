"""The agent loop.

    result = await run(agent, "What's the weather in Paris?")

Each turn: build the prompt, call the model, run any requested tools (in
parallel), feed results back, repeat until the model answers without tool
calls. Handoffs switch the active agent. Limits, cancellation and guardrails
are checked at every step.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .agent import Agent, Handoff
from .context import CancelToken, RunContext, RunLimits
from .errors import (
    GuardrailTripped,
    MaxToolCallsExceeded,
    MaxTurnsExceeded,
    OpenHarnessError,
    OutputValidationError,
    RunCancelled,
    RunStopped,
    RunTimeout,
    TokenBudgetExceeded,
    ToolError,
)
from .guardrails import Guardrail
from .memory import memory_tools
from .models import Model
from .models.base import ModelEvent
from .result import RunResult, RunStream
from .schema import SchemaValidationError, output_schema_for, parse_output, schema_name
from .sessions import Session
from .tools import Tool
from .tracing import Trace, TraceProcessor, global_processors
from .types import Message, ModelRequest, ModelResponse, ModelSettings, OutputSchema, StreamEvent, ToolCall, to_json

Approver = Callable[[RunContext, ToolCall], "bool | Awaitable[bool]"]
EventHook = Callable[[StreamEvent], "None | Awaitable[None]"]


class _Deadline:
    def __init__(self, limits: RunLimits, cancel: CancelToken):
        self.limits = limits
        self.cancel = cancel
        self.start = time.monotonic()

    def remaining(self) -> float | None:
        if self.limits.timeout_seconds is None:
            return None
        return self.limits.timeout_seconds - (time.monotonic() - self.start)

    async def guard(self, awaitable: Awaitable[Any]) -> Any:
        """Await something, aborting on cancellation or when the run's time is up."""
        task = asyncio.ensure_future(awaitable)
        stopper = asyncio.ensure_future(self.cancel.wait())
        remaining = self.remaining()
        try:
            done, _ = await asyncio.wait(
                {task, stopper},
                timeout=None if remaining is None else max(remaining, 0),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            stopper.cancel()
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        if self.cancel.cancelled:
            raise RunCancelled(f"Run cancelled: {self.cancel.reason}")
        raise RunTimeout(f"Run exceeded timeout of {self.limits.timeout_seconds}s")


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _input_messages(value: str | Message | list[Message]) -> list[Message]:
    if isinstance(value, str):
        return [Message.user(value)]
    if isinstance(value, Message):
        return [value]
    return list(value)


async def _run_guardrails(guardrails: list[Guardrail], ctx: RunContext, value: Any, stage: str, trace: Trace) -> Any:
    for g in guardrails:
        with trace.span("guardrail", g.name, stage=stage) as sp:
            res = await g.check(ctx, value)
            sp.set(tripped=res.tripwire)
        if res.tripwire:
            raise GuardrailTripped(f"{stage} guardrail '{g.name}' tripped", g.name, stage, res.info)
        if res.replacement is not None:
            value = res.replacement
    return value


async def _stream_loop(
    agent: Agent,
    input: str | Message | list[Message],
    *,
    session: Session | None = None,
    deps: Any = None,
    model: str | Model | None = None,
    limits: RunLimits | None = None,
    model_settings: ModelSettings | None = None,
    cancel_token: CancelToken | None = None,
    approve: Approver | None = None,
    trace: TraceProcessor | list[TraceProcessor] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    streaming: bool = True,
    _parent_trace: Trace | None = None,
) -> AsyncIterator[StreamEvent]:
    limits = limits or agent.limits or RunLimits()
    cancel = cancel_token or CancelToken()
    deadline = _Deadline(limits, cancel)
    procs: list[TraceProcessor] = trace if isinstance(trace, list) else ([trace] if trace else [])
    tr = _parent_trace or Trace(agent.name, [*procs, *global_processors()], trace_metadata)
    owns_trace = _parent_trace is None

    history = await session.get_messages() if session else []
    inputs = _input_messages(input)
    ctx: RunContext = RunContext(agent=agent, deps=deps, trace=tr, cancel_token=cancel)
    new: list[Message] = []
    current = agent
    output_retries = 0
    mcp_cache: dict[int, list[Tool]] = {}

    def partial(text: str = "") -> RunResult:
        return RunResult(None, text, new, current, ctx.usage, ctx.turn, ctx.tool_calls, tr, history)

    def event(type_: str, **data: Any) -> StreamEvent:
        return StreamEvent(type_, data, agent=current.name)

    run_span_cm = tr.span("run" if owns_trace else "agent_tool", agent.name)
    run_span = run_span_cm.__enter__()
    try:
        # ---- input guardrails (on the latest user text)
        if agent.input_guardrails:
            idx = next((i for i in range(len(inputs) - 1, -1, -1) if inputs[i].role == "user"), None)
            if idx is not None:
                replaced = await deadline.guard(
                    _run_guardrails(agent.input_guardrails, ctx, inputs[idx].content, "input", tr)
                )
                if replaced != inputs[idx].content:
                    inputs[idx] = Message.user(str(replaced))
        new.extend(inputs)
        yield event("agent_start", agent=current.name)

        while True:
            # ---- limits
            if cancel.cancelled:
                raise RunCancelled(f"Run cancelled: {cancel.reason}")
            if limits.max_turns is not None and ctx.turn >= limits.max_turns:
                raise MaxTurnsExceeded(f"Agent stopped after reaching max_turns={limits.max_turns}")
            if limits.max_total_tokens is not None and ctx.usage.total_tokens >= limits.max_total_tokens:
                raise TokenBudgetExceeded(
                    f"Token budget of {limits.max_total_tokens} used up ({ctx.usage.total_tokens} tokens)"
                )
            rem = deadline.remaining()
            if rem is not None and rem <= 0:
                raise RunTimeout(f"Run exceeded timeout of {limits.timeout_seconds}s")
            ctx.turn += 1
            ctx.agent = current

            # ---- assemble tools
            tools: list[Tool] = list(current.tools)
            if current.memory is not None:
                tools += memory_tools(current.memory)
            for server in current.mcp_servers:
                key = id(server)
                if key not in mcp_cache:
                    with tr.span("mcp", server.name, action="list_tools"):
                        mcp_cache[key] = await deadline.guard(server.list_tools())
                tools += mcp_cache[key]
            handoffs = {h.name: h for h in current.handoff_objects()}
            tools += [h.tool() for h in handoffs.values()]
            by_name: dict[str, Tool] = {}
            for t in tools:
                by_name.setdefault(t.name, t)

            # ---- system prompt
            system = await deadline.guard(_maybe_await(current.render_instructions(ctx)))
            if current.memory is not None:
                query = next((m.content for m in reversed([*history, *new]) if m.role == "user"), "")
                found = await current.memory.search(query, limit=5)
                if found:
                    system += "\n\nRelevant long-term memories:\n" + "\n".join(f"- {m.text}" for m in found)

            m = current.get_model(model)
            settings = current.model_settings.merged(model_settings)
            out_schema = None
            if current.output_type is not None:
                schema = output_schema_for(current.output_type)
                native = not (m.provider == "gemini" and by_name)  # Gemini: schema + tools not combined
                if native:
                    out_schema = OutputSchema(schema_name(current.output_type), schema)
                else:
                    system += (
                        "\n\nWhen you give your final answer, reply with only a JSON object that matches "
                        f"this JSON Schema:\n{json.dumps(schema)}"
                    )

            request = ModelRequest(
                system=system,
                messages=[*history, *new],
                tools=[t.spec() for t in by_name.values()],
                output_schema=out_schema,
                settings=settings,
            )

            # ---- call the model
            response: ModelResponse | None = None
            with tr.span("model", f"{m.provider}:{m.model}", agent=current.name, turn=ctx.turn) as msp:
                if streaming:
                    gen = m.stream(request).__aiter__()
                    while True:
                        try:
                            ev: ModelEvent = await deadline.guard(gen.__anext__())
                        except StopAsyncIteration:
                            break
                        if ev.type == "text_delta":
                            yield event("text_delta", delta=ev.data["delta"])
                        elif ev.type == "done":
                            response = ev.response
                    if response is None:
                        raise OpenHarnessError(f"{m!r} stream ended without a response")
                else:
                    response = await deadline.guard(m.generate(request))
                msp.set(
                    usage=response.usage.to_dict(),
                    stop_reason=response.stop_reason,
                    tool_calls=[tc.name for tc in response.message.tool_calls],
                )
            ctx.usage.add(response.usage)
            msg = response.message
            msg.name = current.name
            new.append(msg)
            ctx.messages = [*history, *new]
            yield event("message", message=msg)

            # ---- tool calls
            if msg.tool_calls:
                calls = msg.tool_calls
                if limits.max_tool_calls is not None and ctx.tool_calls + len(calls) > limits.max_tool_calls:
                    new.pop()  # keep history valid: drop the unanswered tool request
                    raise MaxToolCallsExceeded(f"Agent stopped after reaching max_tool_calls={limits.max_tool_calls}")
                ctx.tool_calls += len(calls)
                handoff_target: Handoff | None = None
                for tc in calls:
                    yield event("tool_call", id=tc.id, name=tc.name, arguments=tc.arguments)

                async def execute(tc: ToolCall, by_name: dict[str, Tool] = by_name) -> Message:
                    nonlocal handoff_target
                    t = by_name.get(tc.name)
                    if t is None:
                        return Message.tool(
                            tc.id,
                            tc.name,
                            f"Error: unknown tool '{tc.name}'. Available: {', '.join(by_name) or 'none'}",
                            is_error=True,
                        )
                    h = t.metadata.get("handoff")
                    if h is not None:
                        if handoff_target is None:
                            handoff_target = h
                            return Message.tool(tc.id, tc.name, f"Transferred to {h.agent.name}.")
                        return Message.tool(tc.id, tc.name, "Ignored: another handoff was already made.", is_error=True)
                    with tr.span("tool", tc.name, parent=msp, arguments=tc.arguments) as tsp:
                        if t.requires_approval(tc.arguments):
                            ok = await _maybe_await(approve(ctx, tc)) if approve else False
                            tsp.set(approved=ok)
                            if not ok:
                                return Message.tool(
                                    tc.id, tc.name, "Error: the user did not approve this tool call.", is_error=True
                                )
                        try:
                            value = await t.invoke(tc.arguments, ctx)
                            out = to_json(value)
                            tsp.set(output=out[:2000])
                            return Message.tool(tc.id, tc.name, out)
                        except (RunStopped, asyncio.CancelledError):
                            raise
                        except ToolError as e:
                            tsp.error = str(e)
                            return Message.tool(tc.id, tc.name, f"Error: {e}", is_error=True)
                        except Exception as e:  # tool bugs go back to the model, not up the stack
                            tsp.error = f"{type(e).__name__}: {e}"
                            return Message.tool(tc.id, tc.name, f"Error: {type(e).__name__}: {e}", is_error=True)

                results = await deadline.guard(asyncio.gather(*(execute(tc) for tc in calls)))
                for r in results:
                    new.append(r)
                    yield event("tool_result", id=r.tool_call_id, name=r.name, output=r.content, is_error=r.is_error)
                if handoff_target is not None:
                    prev = current
                    with tr.span("handoff", f"{prev.name} -> {handoff_target.agent.name}"):
                        current = handoff_target.agent
                        if handoff_target.input_filter:
                            kept = handoff_target.input_filter([*history, *new])
                            history, new = [], list(kept)
                    yield event("handoff", **{"from": prev.name, "to": current.name})
                    yield event("agent_start", agent=current.name)
                continue

            # ---- final answer
            text = msg.content
            output: Any = text
            if current.output_type is not None:
                try:
                    output = parse_output(text, current.output_type, output_schema_for(current.output_type))
                except (SchemaValidationError, ValueError, TypeError) as e:
                    if output_retries >= limits.max_output_retries:
                        raise OutputValidationError(f"Structured output failed validation: {e}") from e
                    output_retries += 1
                    new.append(
                        Message.user(
                            f"Your reply did not match the required JSON schema: {e}. "
                            "Reply again with only the corrected JSON."
                        )
                    )
                    continue
            if current.output_guardrails:
                output = await deadline.guard(_run_guardrails(current.output_guardrails, ctx, output, "output", tr))
                if isinstance(output, str):
                    text = output
            result = RunResult(output, text, new, current, ctx.usage, ctx.turn, ctx.tool_calls, tr, history)
            if session is not None:
                await session.add_messages(new)
            run_span.set(turns=ctx.turn, usage=ctx.usage.to_dict(), last_agent=current.name)
            yield event("run_end", result=result)
            return
    except RunStopped as e:
        if e.partial is None:
            e.partial = partial()
        run_span.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        run_span_cm.__exit__(None, None, None)
        if owns_trace:
            tr.finish()


async def run(agent: Agent, input: str | Message | list[Message], **kwargs: Any) -> RunResult:
    """Run an agent to completion and return the result.

    Keyword args: session, deps, model, limits, model_settings, cancel_token,
    approve, trace, trace_metadata, on_event.
    """
    on_event: EventHook | None = kwargs.pop("on_event", None)
    result: RunResult | None = None
    async for ev in _stream_loop(agent, input, streaming=on_event is not None, **kwargs):
        if on_event is not None:
            await _maybe_await(on_event(ev))
        if ev.type == "run_end":
            result = ev.data["result"]
    assert result is not None
    return result


def run_stream(agent: Agent, input: str | Message | list[Message], **kwargs: Any) -> RunStream:
    """Start a run and stream events. See ``RunStream``."""
    token = kwargs.pop("cancel_token", None) or CancelToken()
    return RunStream(_stream_loop(agent, input, cancel_token=token, streaming=True, **kwargs), token)


def run_sync(agent: Agent, input: str | Message | list[Message], **kwargs: Any) -> RunResult:
    """Blocking version of ``run`` for scripts and notebooks without an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run(agent, input, **kwargs))
    raise OpenHarnessError("run_sync() cannot be used inside a running event loop; use 'await run(...)'")
