# open-harness for TypeScript

A self-hosted SDK for building AI agents: tools, structured output, sessions, memory, streaming, guardrails, limits,
tracing, MCP and multi-agent, on Anthropic, OpenAI, Gemini or any OpenAI-compatible model. Zero runtime dependencies.
Node.js 20+.

```bash
npm install openharness
```

```ts
import { Agent, run, s, tool } from "openharness";

const getWeather = tool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: s.object({ city: s.string() }),
  execute: ({ city }) => `18C and cloudy in ${city}`,
});

const agent = new Agent({ instructions: "Be brief.", model: "anthropic:claude-sonnet-5", tools: [getWeather] });
console.log((await run(agent, "Do I need a jacket in Toronto?")).output);
```

Or the ready-made harness and CLI:

```ts
import { Harness } from "openharness";
console.log(await new Harness().ask("What is 17.5% of 2,340?"));
```

```bash
npx openharness chat --model ollama:llama3.2
```

Documentation: https://nazsam.github.io/open-harness/ · Source: https://github.com/nazsam/open-harness
