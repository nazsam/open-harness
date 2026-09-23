import asyncio
from dataclasses import dataclass
from typing import Literal

import pytest

from openharness import (
    Agent,
    CancelToken,
    FakeModel,
    FileMemory,
    FileSession,
    GuardrailTripped,
    InMemoryMemory,
    InMemorySession,
    MaxToolCallsExceeded,
    MaxTurnsExceeded,
    MemoryProcessor,
    OutputValidationError,
    RunCancelled,
    RunContext,
    RunLimits,
    RunTimeout,
    SQLiteSession,
    TokenBudgetExceeded,
    ToolError,
    call,
    input_guardrail,
    max_length,
    output_guardrail,
    pii,
    run,
    run_stream,
    run_sync,
    tool,
)
from openharness.guardrails import GuardrailResult


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@dataclass
class Weather:
    city: str
    temp_c: float
    conditions: Literal["sunny", "cloudy", "rain"]


# ------------------------------------------------------------------ tools


def test_tool_schema_from_hints_and_docstring():
    spec = add.spec()
    assert spec.name == "add"
    assert spec.description == "Add two integers."
    assert spec.parameters["properties"]["a"] == {"type": "integer", "description": "First number."}
    assert spec.parameters["required"] == ["a", "b"]
    assert spec.parameters["additionalProperties"] is False


def test_tool_schema_optional_literal_dataclass_and_context():
    @tool
    def search(
        ctx: RunContext, query: str, limit: int = 5, kind: Literal["a", "b"] | None = None, where: Weather | None = None
    ) -> str:
        """Search."""
        return ""

    p = search.parameters
    assert "ctx" not in p["properties"]
    assert p["required"] == ["query"]
    assert p["properties"]["limit"]["default"] == 5
    assert {"enum": ["a", "b"], "type": "string"} in p["properties"]["kind"]["anyOf"]
    weather = p["properties"]["where"]["anyOf"][0]
    assert weather["required"] == ["city", "temp_c", "conditions"]


async def test_basic_tool_loop_and_usage():
    model = FakeModel([call("add", a=2, b=40), "The answer is 42."])
    agent = Agent(model=model, tools=[add])
    result = await run(agent, "What is 2 + 40?")
    assert result.output == "The answer is 42."
    assert result.turns == 2 and result.tool_calls == 1
    roles = [m.role for m in result.new_messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert result.new_messages[2].content == "42"
    assert result.usage.requests == 2
    # the second request carried the tool result back to the model
    assert model.requests[1].messages[-1].role == "tool"


async def test_parallel_tools_run_concurrently():
    started = []

    @tool
    async def slow(n: int) -> int:
        """Slow."""
        started.append(n)
        await asyncio.sleep(0.2)
        return n

    model = FakeModel([{"tool_calls": [{"name": "slow", "arguments": {"n": i}} for i in range(5)]}, "done"])
    t0 = asyncio.get_running_loop().time()
    result = await run(Agent(model=model, tools=[slow]), "go")
    assert asyncio.get_running_loop().time() - t0 < 0.8
    assert [m.content for m in result.new_messages if m.role == "tool"] == ["0", "1", "2", "3", "4"]


async def test_tool_errors_go_back_to_model():
    @tool
    def broken(x: int) -> int:
        """Broken."""
        raise ToolError("x is not allowed")

    model = FakeModel([call("broken", x=1), call("nope"), call("add", a="oops", b=1), "ok"])
    result = await run(Agent(model=model, tools=[broken, add]), "go")
    errors = [m for m in result.new_messages if m.role == "tool"]
    assert all(m.is_error for m in errors)
    assert "x is not allowed" in errors[0].content
    assert "unknown tool 'nope'" in errors[1].content
    assert "Invalid arguments" in errors[2].content


async def test_plain_functions_become_tools_and_context_injection():
    def whoami(ctx: RunContext) -> str:
        """Return the user."""
        return ctx.deps["user"]

    model = FakeModel([call("whoami"), "hi"])
    result = await run(Agent(model=model, tools=[whoami]), "who", deps={"user": "sam"})
    assert result.new_messages[2].content == "sam"


# ------------------------------------------------------------------ structured output


async def test_structured_output_dataclass():
    model = FakeModel([{"json": {"city": "Paris", "temp_c": 21, "conditions": "sunny"}}])
    result = await run(Agent(model=model, output_type=Weather), "weather?")
    assert result.output == Weather("Paris", 21.0, "sunny")
    assert model.requests[0].output_schema.schema["required"] == ["city", "temp_c", "conditions"]


async def test_structured_output_retries_then_succeeds():
    model = FakeModel(
        [
            "not json",
            {"json": {"city": "Oslo", "temp_c": 3, "conditions": "snow"}},
            '```json\n{"city": "Oslo", "temp_c": 3, "conditions": "rain"}\n```',
        ]
    )
    result = await run(Agent(model=model, output_type=Weather), "weather?")
    assert result.output.conditions == "rain"
    assert result.turns == 3


async def test_structured_output_gives_up():
    model = FakeModel(default="nope")
    with pytest.raises(OutputValidationError):
        await run(Agent(model=model, output_type=Weather), "weather?", limits=RunLimits(max_output_retries=1))


async def test_structured_output_list_and_pydantic():
    pydantic = pytest.importorskip("pydantic")

    class Item(pydantic.BaseModel):
        name: str
        qty: int

    model = FakeModel([{"json": {"value": [{"name": "a", "qty": 1}]}}])
    result = await run(Agent(model=model, output_type=list[Item]), "list")
    assert result.output == [Item(name="a", qty=1)]


# ------------------------------------------------------------------ streaming


async def test_streaming_events():
    model = FakeModel([call("add", a=1, b=1), "Two it is."], chunk_size=3)
    stream = run_stream(Agent(model=model, tools=[add]), "1+1")
    types, text = [], ""
    async for ev in stream:
        types.append(ev.type)
        if ev.type == "text_delta":
            text += ev.data["delta"]
    result = await stream.result()
    assert text == "Two it is." == result.output
    assert types[0] == "agent_start" and types[-1] == "run_end"
    assert types.index("tool_call") < types.index("tool_result") < types.index("text_delta")


# ------------------------------------------------------------------ sessions and memory


@pytest.mark.parametrize("kind", ["memory", "file", "sqlite"])
async def test_sessions_remember_history(kind, tmp_path):
    session = {
        "memory": InMemorySession(),
        "file": FileSession(tmp_path / "s.jsonl"),
        "sqlite": SQLiteSession("alice", tmp_path / "db.sqlite"),
    }[kind]
    model = FakeModel(["Nice to meet you", lambda req: f"seen {len(req.messages)} messages"])
    agent = Agent(model=model)
    await run(agent, "I am Alice", session=session)
    result = await run(agent, "who am I?", session=session)
    assert result.output == "seen 3 messages"
    assert len(await session.get_messages()) == 4
    await session.clear()
    assert await session.get_messages() == []


async def test_memory_tools_and_injection(tmp_path):
    mem = FileMemory(tmp_path / "mem.json")
    model = FakeModel([call("remember", fact="The user prefers metric units"), "Saved.", "ok"])
    agent = Agent(model=model, memory=mem)
    await run(agent, "remember I like metric")
    assert [m.text for m in await FileMemory(tmp_path / "mem.json").list()] == ["The user prefers metric units"]
    await run(agent, "what units do I prefer?")
    assert "prefers metric units" in model.requests[-1].system
    assert [t.name for t in model.requests[-1].tools] == ["remember", "recall"]


async def test_memory_search_ranking():
    mem = InMemoryMemory()
    await mem.add("Sam lives in Toronto")
    await mem.add("Favourite language is Python")
    await mem.add("Sam lives in Toronto")  # duplicate ignored
    hits = await mem.search("which city does Sam live in? Toronto")
    assert hits[0].text == "Sam lives in Toronto" and len(await mem.list()) == 2


# ------------------------------------------------------------------ guardrails


async def test_input_guardrail_blocks():
    model = FakeModel(["never"])
    with pytest.raises(GuardrailTripped) as e:
        await run(Agent(model=model, input_guardrails=[max_length(5)]), "this is too long")
    assert e.value.stage == "input" and model.requests == []


async def test_pii_redaction_and_output_guardrail():
    @output_guardrail
    def no_secrets(ctx, out):
        return GuardrailResult(tripwire="SECRET" in out)

    model = FakeModel(["Got it", "the SECRET is 42"])
    agent = Agent(model=model, input_guardrails=[pii()], output_guardrails=[no_secrets])
    await run(agent, "mail me at sam@example.com or 416-555-0199")
    assert model.requests[0].messages[0].content == "mail me at [EMAIL] or [PHONE]"
    with pytest.raises(GuardrailTripped):
        await run(agent, "tell me")


async def test_async_custom_input_guardrail():
    @input_guardrail
    async def no_homework(ctx, text):
        return "homework" in text

    with pytest.raises(GuardrailTripped):
        await run(Agent(model=FakeModel(), input_guardrails=[no_homework]), "do my homework")


# ------------------------------------------------------------------ limits and cancellation


async def test_max_turns():
    model = FakeModel(default=None, script=[call("add", a=1, b=1)] * 10)
    with pytest.raises(MaxTurnsExceeded) as e:
        await run(Agent(model=model, tools=[add]), "loop", limits=RunLimits(max_turns=3))
    assert e.value.partial.turns == 3


async def test_max_tool_calls():
    model = FakeModel([call("add", a=1, b=1)] * 10)
    with pytest.raises(MaxToolCallsExceeded):
        await run(Agent(model=model, tools=[add]), "loop", limits=RunLimits(max_tool_calls=2))


async def test_token_budget():
    model = FakeModel([call("add", a=1, b=1)] * 10)
    with pytest.raises(TokenBudgetExceeded):
        await run(Agent(model=model, tools=[add]), "loop", limits=RunLimits(max_total_tokens=25))


async def test_timeout_interrupts_slow_tool():
    @tool
    async def sleepy() -> str:
        """Sleep."""
        await asyncio.sleep(10)
        return "late"

    with pytest.raises(RunTimeout):
        await run(Agent(model=FakeModel([call("sleepy")]), tools=[sleepy]), "go", limits=RunLimits(timeout_seconds=0.3))


async def test_cancel_token_stops_run():
    @tool
    async def sleepy() -> str:
        """Sleep."""
        await asyncio.sleep(10)
        return "late"

    token = CancelToken()
    asyncio.get_running_loop().call_later(0.2, token.cancel, "user pressed stop")
    with pytest.raises(RunCancelled, match="user pressed stop"):
        await run(Agent(model=FakeModel([call("sleepy")]), tools=[sleepy]), "go", cancel_token=token)


# ------------------------------------------------------------------ approvals


async def test_tool_approval():
    @tool(needs_approval=True)
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"deleted {path}"

    decisions = []

    def approve(ctx, tc):
        decisions.append(tc.arguments["path"])
        return tc.arguments["path"] == "tmp.txt"

    model = FakeModel([call("delete_file", path="tmp.txt"), call("delete_file", path="/etc/passwd"), "done"])
    result = await run(Agent(model=model, tools=[delete_file]), "clean", approve=approve)
    outs = [m for m in result.new_messages if m.role == "tool"]
    assert outs[0].content == "deleted tmp.txt"
    assert outs[1].is_error and "did not approve" in outs[1].content
    assert decisions == ["tmp.txt", "/etc/passwd"]


# ------------------------------------------------------------------ multi-agent


async def test_handoff_switches_agent():
    billing_model = FakeModel(["Your refund is on its way."])
    billing = Agent(name="Billing", description="Handles refunds.", model=billing_model)
    triage_model = FakeModel([call("transfer_to_billing")])
    triage = Agent(name="Triage", model=triage_model, handoffs=[billing])
    result = await run(triage, "I want a refund")
    assert result.output == "Your refund is on its way."
    assert result.last_agent is billing
    assert [t.name for t in triage_model.requests[0].tools] == ["transfer_to_billing"]
    assert billing_model.requests[0].messages[0].content == "I want a refund"


async def test_agent_as_tool():
    researcher = Agent(name="Researcher", model=FakeModel(["Paris is the capital of France."]))
    lead = Agent(
        name="Lead",
        model=FakeModel([call("researcher", input="capital of France?"), "It is Paris."]),
        tools=[researcher.as_tool()],
    )
    result = await run(lead, "What is the capital of France?")
    assert result.output == "It is Paris."
    assert result.new_messages[2].content == "Paris is the capital of France."
    assert result.usage.requests == 3


# ------------------------------------------------------------------ tracing and sync API


async def test_tracing_spans():
    proc = MemoryProcessor()
    await run(Agent(model=FakeModel([call("add", a=1, b=2), "3"]), tools=[add]), "1+2", trace=proc)
    trace = proc.traces[0]
    kinds = [s.kind for s in trace.spans]
    assert kinds.count("model") == 2 and kinds.count("tool") == 1 and kinds[0] == "run"
    tool_span = next(s for s in trace.spans if s.kind == "tool")
    model_span = next(s for s in trace.spans if s.kind == "model")
    assert tool_span.parent_id == model_span.id and tool_span.attributes["output"] == "3"
    assert all(s.end is not None for s in trace.spans)


def test_run_sync():
    assert run_sync(Agent(model=FakeModel(["hello"])), "hi").output == "hello"


async def test_dynamic_instructions():
    async def instructions(ctx):
        return f"User is {ctx.deps}."

    model = FakeModel(["ok"])
    await run(Agent(model=model, instructions=instructions), "hi", deps="Sam")
    assert model.requests[0].system == "User is Sam."
