---
title: CLI
parent: Guides
nav_order: 11
---

# CLI

Both SDKs install the same `openharness` command.

```bash
openharness chat --model anthropic:claude-sonnet-5
openharness chat --model ollama:llama3.2 --session chats/today.jsonl --memory memory.json
openharness run "Summarise this" < notes.txt
cat error.log | openharness run "What went wrong?" --model openai:gpt-6-luna
openharness models
```

## Chat commands

| Command | |
|---|---|
| `/reset` | Clear the conversation |
| `/usage` | Token usage so far |
| `/tools` | List tools, including MCP tools |
| `/exit` | Quit (Ctrl+D also works) |

Press **Ctrl+C** while the agent is replying to stop that reply without leaving the chat. Tool calls and results are
shown as they happen.

## Options

| Option | |
|---|---|
| `-m, --model` | `provider:model` (default: from `OPENHARNESS_MODEL` or your API keys) |
| `-s, --system` | System instructions |
| `--session FILE` | Save the conversation to a `.jsonl` file and resume it later |
| `--memory FILE` | Long-term memory file (adds `remember` and `recall`) |
| `--tools LIST` | `calculator,current_time,fetch_url` or `none` (default: `calculator,current_time`) |
| `--mcp CMD` | MCP stdio server command, repeatable |
| `--mcp-http URL` | MCP HTTP server, repeatable |
| `--max-turns N`, `--max-tool-calls N`, `--timeout S` | Limits per reply |
| `--trace console` or `--trace FILE` | Print spans or write JSON Lines traces |
| `-y, --yes` | Approve tools that need approval without asking |

Exit codes: `0` success, `1` error, `2` a limit or guardrail stopped the run.
