# open-harness for Python

A self-hosted SDK for building AI agents: tools, structured output, sessions, memory, streaming, guardrails, limits,
tracing, MCP and multi-agent, on Anthropic, OpenAI, Gemini or any OpenAI-compatible model. Only dependency: `httpx`.

```bash
pip install open-harness
```

```python
from openharness import Agent, run_sync, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and cloudy in {city}"

agent = Agent(instructions="Be brief.", model="anthropic:claude-sonnet-5", tools=[get_weather])
print(run_sync(agent, "Do I need a jacket in Toronto?").output)
```

Or the ready-made harness and CLI:

```python
from openharness import Harness
print(Harness().ask_sync("What is 17.5% of 2,340?"))
```

```bash
openharness chat --model ollama:llama3.2
```

Documentation: https://nazsam.github.io/open-harness/ · Source: https://github.com/nazsam/open-harness
