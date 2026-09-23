"""Stream text and tool activity as it happens. Press Ctrl+C to stop mid-reply."""

import asyncio

from openharness import Agent, calculator, run_stream


async def main() -> None:
    agent = Agent(instructions="Explain your reasoning briefly.", tools=[calculator])
    stream = run_stream(agent, "If I save $350 a month at 4% a year, roughly how much after 5 years?")
    try:
        async for event in stream:
            if event.type == "text_delta":
                print(event.data["delta"], end="", flush=True)
            elif event.type == "tool_call":
                print(f"\n[calling {event.data['name']} {event.data['arguments']}]")
    except KeyboardInterrupt:
        stream.cancel()
    result = await stream.result()
    print(f"\n\n{result.usage.to_dict()}")


asyncio.run(main())
