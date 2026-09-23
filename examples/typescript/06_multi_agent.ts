// Two multi-agent patterns: handoffs (triage) and agents as tools (manager + workers).
import { Agent, run } from "openharness";

// handoffs: the triage agent passes the whole conversation to a specialist
const billing = new Agent({ name: "Billing", description: "Refunds, invoices and payments.", instructions: "You handle billing questions. Be precise about amounts." });
const tech = new Agent({ name: "Tech support", description: "Bugs, errors and setup help.", instructions: "You troubleshoot technical problems step by step." });
const triage = new Agent({ name: "Triage", instructions: "Route the user to the right specialist. Do not answer yourself.", handoffs: [billing, tech] });

// agents as tools: a manager calls workers and combines their answers
const researcher = new Agent({ name: "Researcher", instructions: "Give 3 short, factual bullet points on the topic." });
const critic = new Agent({ name: "Critic", instructions: "List the 2 biggest risks or weaknesses of the idea." });
const manager = new Agent({
  name: "Manager",
  instructions: "Use the researcher and the critic, then write a 5 sentence recommendation.",
  tools: [researcher.asTool(), critic.asTool()],
});

const r1 = await run(triage, "I was charged twice for my subscription this month.");
console.log(`[${r1.lastAgent.name}] ${r1.output}\n`);
console.log((await run(manager, "Should a 10 person startup self-host its LLM agents?")).output);
