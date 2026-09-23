"""Tools plus typed, validated output."""

import asyncio
from dataclasses import dataclass
from typing import Literal

from openharness import Agent, run, tool


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Toronto".
    """
    fake = {"toronto": (14, "cloudy"), "paris": (21, "sunny"), "london": (11, "rain")}
    temp, cond = fake.get(city.lower(), (18, "cloudy"))
    return {"city": city, "temp_c": temp, "conditions": cond}


@dataclass
class TripAdvice:
    city: str
    temp_c: float
    conditions: Literal["sunny", "cloudy", "rain"]
    pack: list[str]


agent = Agent(
    name="Travel helper",
    instructions="Check the weather with your tool, then suggest what to pack.",
    tools=[get_weather],
    output_type=TripAdvice,
)


async def main() -> None:
    result = await run(agent, "I'm flying to London tomorrow.")
    advice: TripAdvice = result.output
    print(advice)
    print(f"{result.turns} turns, {result.usage.total_tokens} tokens")


asyncio.run(main())
