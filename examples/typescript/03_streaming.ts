// Stream text and tool activity as it happens. Press Ctrl+C to stop mid-reply.
import { Agent, calculator, runStream } from "openharness";

const agent = new Agent({ instructions: "Explain your reasoning briefly.", tools: [calculator] });
const stream = runStream(agent, "If I save $350 a month at 4% a year, roughly how much after 5 years?");
process.once("SIGINT", () => stream.cancel("interrupted"));

for await (const event of stream) {
  if (event.type === "text_delta") process.stdout.write(event.data.delta);
  else if (event.type === "tool_call") console.log(`\n[calling ${event.data.name} ${JSON.stringify(event.data.arguments)}]`);
}
const result = await stream.result();
console.log("\n\n", result.usage.toJSON());
