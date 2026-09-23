---
title: Examples
nav_order: 4
---

# Examples

Runnable examples live in [`examples/`](https://github.com/nazsam/open-harness/tree/main/examples). Every example exists
in both languages with the same number.

| # | Example | Shows |
|---|---|---|
| 01 | [Quickstart](https://github.com/nazsam/open-harness/blob/main/examples/python/01_quickstart.py) | The harness in two lines |
| 02 | [Tools and structured output](https://github.com/nazsam/open-harness/blob/main/examples/python/02_tools_structured_output.py) | A tool plus a typed result |
| 03 | [Streaming](https://github.com/nazsam/open-harness/blob/main/examples/python/03_streaming.py) | Live text and tool events, Ctrl+C to stop |
| 04 | [Sessions and memory](https://github.com/nazsam/open-harness/blob/main/examples/python/04_sessions_and_memory.py) | Two conversations sharing long-term memory |
| 05 | [Guardrails and limits](https://github.com/nazsam/open-harness/blob/main/examples/python/05_guardrails_and_limits.py) | PII redaction, blocked patterns, approvals, budgets |
| 06 | [Multi-agent](https://github.com/nazsam/open-harness/blob/main/examples/python/06_multi_agent.py) | Triage handoffs and a manager with worker agents |
| 07 | [MCP](https://github.com/nazsam/open-harness/blob/main/examples/python/07_mcp.py) | The filesystem MCP server as agent tools |
| 08 | [Switch providers](https://github.com/nazsam/open-harness/blob/main/examples/python/08_switch_providers.py) | One agent on Claude, OpenAI, Gemini and Ollama |
| 09 | [Offline test](https://github.com/nazsam/open-harness/blob/main/examples/python/09_offline_test.py) | Testing with `FakeModel`, no API key |

## Running them

```bash
git clone https://github.com/nazsam/open-harness && cd open-harness
export ANTHROPIC_API_KEY=...            # or any provider, or OPENHARNESS_MODEL=ollama:llama3.2

# Python
pip install -e python
python examples/python/02_tools_structured_output.py

# TypeScript
(cd typescript && npm install && npm run build)
cd examples/typescript && npm install
npx tsx 02_tools_structured_output.ts
```

`OPENHARNESS_MODEL=fake:demo` runs any example without a key (the fake model just echoes, so it only checks the wiring).
