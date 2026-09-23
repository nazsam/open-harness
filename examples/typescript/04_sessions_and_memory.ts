// Sessions remember the conversation. Memory remembers facts across conversations.
import { Agent, FileMemory, FileSession, run } from "openharness";

const memory = new FileMemory("data/memory.json"); // durable facts, shared by every session
const agent = new Agent({ instructions: "You are a personal assistant. Save lasting preferences with remember.", memory });

const monday = new FileSession("data/monday.jsonl");
await run(agent, "Hi, I'm Sam. I always want temperatures in Celsius.", { session: monday });
console.log((await run(agent, "What's my name?", { session: monday })).output);

const tuesday = new FileSession("data/tuesday.jsonl"); // new conversation, same memory
console.log((await run(agent, "Which temperature unit do I prefer?", { session: tuesday })).output);
console.log("memories:", (await memory.list()).map((m) => m.text));
