"""Live smoke test against a real provider. Run by the "Live test" workflow.

    OPENHARNESS_MODEL=anthropic:claude-sonnet-5 ANTHROPIC_API_KEY=... python scripts/live_smoke.py
"""

import asyncio
import os
from dataclasses import dataclass

from openharness import Agent, InMemorySession, calculator, run, run_stream

MODEL = os.environ.get("OPENHARNESS_MODEL", "anthropic:claude-sonnet-5")


@dataclass
class Answer:
    result: int
    explanation: str


async def main() -> None:
    # 1. tool call + structured output
    agent = Agent(instructions="Use the calculator for all arithmetic.", model=MODEL, tools=[calculator], output_type=Answer)
    r = await run(agent, "What is 1234 * 5678?")
    assert r.output.result == 7006652, r.output
    assert any(m.role == "tool" for m in r.new_messages), "calculator was not called"
    print(f"[ok] tools + structured output: {r.output.result} ({r.turns} turns, {r.usage.total_tokens} tokens)")

    # 2. streaming with a tool call
    chat = Agent(instructions="Be brief.", model=MODEL, tools=[calculator])
    stream = run_stream(chat, "Use the calculator to compute 2**20, then say the number.")
    text = "".join([d async for d in stream.text()])
    assert "1048576" in text.replace(",", ""), text
    print(f"[ok] streaming: {text.strip()[:80]}")

    # 3. session memory across runs
    session = InMemorySession()
    await run(chat, "My favourite colour is teal. Just say OK.", session=session)
    r = await run(chat, "What is my favourite colour? One word.", session=session)
    assert "teal" in r.output.lower(), r.output
    print(f"[ok] sessions: {r.output.strip()}")
    print("LIVE SMOKE TEST PASSED")


asyncio.run(main())
