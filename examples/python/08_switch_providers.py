"""The same agent on different providers. Only the model string changes."""

import asyncio
import os

from openharness import Agent, calculator, run

MODELS = [
    ("ANTHROPIC_API_KEY", "anthropic:claude-sonnet-5"),
    ("OPENAI_API_KEY", "openai:gpt-6-luna"),
    ("GEMINI_API_KEY", "gemini:gemini-3.8-flash"),
    (None, "ollama:llama3.2"),  # local, no key
]

agent = Agent(instructions="Answer in one sentence. Use the calculator for math.", tools=[calculator])


async def main() -> None:
    for key, model in MODELS:
        if key and not os.environ.get(key):
            continue
        try:
            result = await run(agent, "What is 2**20 divided by 7?", model=model)
            print(f"{model:32} {result.output}")
        except Exception as e:  # e.g. Ollama not running
            print(f"{model:32} skipped: {e}")


asyncio.run(main())
