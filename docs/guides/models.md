---
title: Models and providers
parent: Guides
nav_order: 1
---

# Models and providers
{: .no_toc }

1. TOC
{:toc}

## Model strings

Every model is written as `provider:model`. Pass it to an `Agent`, to `run(..., model=...)` for a single run, to the
`Harness`, or set `OPENHARNESS_MODEL`.

| Provider | Example | API key variable | Notes |
|---|---|---|---|
| `anthropic` | `anthropic:claude-sonnet-5` | `ANTHROPIC_API_KEY` | Messages API. Reasoning blocks are kept across turns. |
| `openai` | `openai:gpt-6-luna` | `OPENAI_API_KEY` | Chat Completions API. `OPENAI_BASE_URL` for proxies and Azure-style gateways. |
| `gemini` | `gemini:gemini-3.8-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `generateContent`. Thought signatures are kept across turns. |
| `ollama` | `ollama:llama3.2` | none | Local, `http://localhost:11434/v1` |
| `lmstudio` | `lmstudio:qwen3` | none | Local, `http://localhost:1234/v1` |
| `vllm` | `vllm:my-model` | none | Local, `http://localhost:8000/v1` |
| `groq` | `groq:<model>` | `GROQ_API_KEY` | OpenAI-compatible |
| `together` | `together:<model>` | `TOGETHER_API_KEY` | OpenAI-compatible |
| `openrouter` | `openrouter:<vendor>/<model>` | `OPENROUTER_API_KEY` | OpenAI-compatible |
| `deepseek` | `deepseek:<model>` | `DEEPSEEK_API_KEY` | OpenAI-compatible |
| `mistral` | `mistral:<model>` | `MISTRAL_API_KEY` | OpenAI-compatible |
| `xai` | `xai:<model>` | `XAI_API_KEY` | OpenAI-compatible |
| `fake` | `fake:demo` | none | Scripted model for tests. See [Testing](testing.html). |

Each local preset's URL can be overridden with `<PROVIDER>_BASE_URL`, for example `OLLAMA_BASE_URL`.

Without a prefix, the provider is inferred from the name (`claude-*`, `gpt-*`, `gemini-*`). With no model at all,
open-harness uses `OPENHARNESS_MODEL`, then the first provider whose API key is set.

Run `openharness models` to print the list with default model names.

{: .note }
Model names change often. The defaults are current as of September 2026; pass any model name your provider supports.

## Switching providers

Only the model string changes. Tools, sessions, guardrails and output types work the same everywhere.

```python
for model in ["anthropic:claude-sonnet-5", "openai:gpt-6-luna", "gemini:gemini-3.8-flash", "ollama:llama3.2"]:
    result = await run(agent, "Summarise our refund policy", model=model)
```

```ts
for (const model of ["anthropic:claude-sonnet-5", "openai:gpt-6-luna", "gemini:gemini-3.8-flash", "ollama:llama3.2"]) {
  const result = await run(agent, "Summarise our refund policy", { model });
}
```

Conversation history is stored in a provider-neutral format, so you can even change provider mid-conversation with the
same session. Provider-only details (such as Claude thinking blocks or Gemini thought signatures) are kept and sent
back to the provider that produced them, and dropped when talking to a different provider.

## Model settings

```python
from openharness import Agent, ModelSettings

agent = Agent(model_settings=ModelSettings(temperature=0.2, max_output_tokens=2000, tool_choice="auto",
                                           parallel_tool_calls=True, extra={"top_k": 40}))
```

```ts
const agent = new Agent({ modelSettings: { temperature: 0.2, maxOutputTokens: 2000, toolChoice: "auto", extra: { top_k: 40 } } });
```

`tool_choice` accepts `"auto"`, `"required"`, `"none"` or a tool name. `extra` is merged into the provider request body
as-is, for any provider-specific parameter.

## Configuring a provider directly

```python
from openharness import AnthropicModel, OpenAIModel

claude = AnthropicModel("claude-sonnet-5", api_key="...", timeout=120, max_retries=5)
company_gateway = OpenAIModel("gpt-6-luna", base_url="https://llm.internal.example.com/v1", api_key="...",
                              headers={"X-Team": "support"})
agent = Agent(model=company_gateway)
```

```ts
import { AnthropicModel, OpenAIModel } from "openharness";

const gateway = new OpenAIModel("gpt-6-luna", { baseUrl: "https://llm.internal.example.com/v1", apiKey: "...", headers: { "X-Team": "support" } });
const agent = new Agent({ model: gateway });
```

Requests are retried on 408, 409, 429 and 5xx with exponential backoff that honours `Retry-After`.

## Bring your own model

Implement `generate` (and optionally `stream`) to plug in anything.

```python
from openharness import Model, Message, Usage
from openharness.types import ModelResponse

class MyModel(Model):
    provider, model = "mine", "v1"

    async def generate(self, request):
        text = await my_llm(request.system, [m.content for m in request.messages])
        return ModelResponse(Message.assistant(text), Usage(requests=1))
```

```ts
import { Message, Usage, type Model } from "openharness";

const myModel: Model = {
  provider: "mine",
  model: "v1",
  async generate(req) {
    return { message: Message.assistant(await myLlm(req)), usage: new Usage({ requests: 1 }) };
  },
  async *stream(req) {
    yield { type: "done", response: await this.generate(req) };
  },
};
```
