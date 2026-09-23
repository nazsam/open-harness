// The same agent on different providers. Only the model string changes.
import { Agent, calculator, run } from "openharness";

const MODELS: [string | null, string][] = [
  ["ANTHROPIC_API_KEY", "anthropic:claude-sonnet-5"],
  ["OPENAI_API_KEY", "openai:gpt-6-luna"],
  ["GEMINI_API_KEY", "gemini:gemini-3.8-flash"],
  [null, "ollama:llama3.2"], // local, no key
];

const agent = new Agent({ instructions: "Answer in one sentence. Use the calculator for math.", tools: [calculator] });
for (const [key, model] of MODELS) {
  if (key && !process.env[key]) continue;
  try {
    const result = await run(agent, "What is 2**20 divided by 7?", { model });
    console.log(model.padEnd(32), result.output);
  } catch (e) {
    console.log(model.padEnd(32), `skipped: ${(e as Error).message}`);
  }
}
