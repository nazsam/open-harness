"""The shortest path to a working agent.

    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY / GEMINI_API_KEY
    python 01_quickstart.py

Or run fully local with Ollama:  OPENHARNESS_MODEL=ollama:llama3.2 python 01_quickstart.py
"""

from openharness import Harness

h = Harness()  # model from the environment, calculator + clock tools, in-memory conversation
print(h.ask_sync("What is 17.5% of 2,340? Then tell me today's date in Toronto."))
