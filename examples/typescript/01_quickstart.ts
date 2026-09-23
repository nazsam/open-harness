// The shortest path to a working agent.
//   export ANTHROPIC_API_KEY=...     (or OPENAI_API_KEY / GEMINI_API_KEY)
//   npx tsx 01_quickstart.ts
// Fully local with Ollama:  OPENHARNESS_MODEL=ollama:llama3.2 npx tsx 01_quickstart.ts
import { Harness } from "openharness";

const h = new Harness(); // model from the environment, calculator + clock tools, in-memory conversation
console.log(await h.ask("What is 17.5% of 2,340? Then tell me today's date in Toronto."));
