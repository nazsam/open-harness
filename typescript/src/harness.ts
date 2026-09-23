/**
 * `Harness`: a working agent in one line, with sensible defaults wired up.
 *
 *   import { Harness } from "openharness";
 *   const h = new Harness();                  // model from env, calculator + clock tools, in-memory session
 *   console.log(await h.ask("What is 17% of 2,340?"));
 *
 * The harness owns an agent, a session, optional long-term memory, MCP
 * connections, limits and tracing. `h.agent` is a normal `Agent`.
 */

import { Agent } from "./agent.js";
import { DEFAULT_TOOLS } from "./builtinTools.js";
import { CancelToken, type RunLimits } from "./context.js";
import type { MCPServer } from "./mcp.js";
import { FileMemory, InMemoryMemory, type Memory } from "./memory.js";
import type { Model } from "./models/index.js";
import type { RunResult } from "./result.js";
import { run, runStream, type Approver, type RunOptions } from "./runner.js";
import { FileSession, InMemorySession, type Session } from "./sessions.js";
import type { Tool } from "./tools.js";
import type { TraceProcessor } from "./tracing.js";
import { Usage, type ModelSettings } from "./types.js";

export const DEFAULT_INSTRUCTIONS =
  "You are a capable, concise assistant. Use tools when they help you give a correct answer, and say so plainly when you are unsure.";

export interface HarnessOptions {
  model?: string | Model;
  instructions?: string;
  tools?: Tool[];
  agent?: Agent<any>;
  /** A Session, a .jsonl path, or "memory" (default). */
  session?: Session | string;
  /** A Memory, a .json path, or true for in-memory. */
  memory?: Memory | string | boolean;
  mcpServers?: MCPServer[];
  limits?: RunLimits;
  modelSettings?: ModelSettings;
  trace?: TraceProcessor | TraceProcessor[];
  approve?: Approver;
  name?: string;
}

export class Harness {
  readonly agent: Agent<any>;
  readonly session: Session;
  limits?: RunLimits;
  trace?: TraceProcessor | TraceProcessor[];
  approve?: Approver;
  readonly usage = new Usage();
  private token?: CancelToken;

  constructor(options: HarnessOptions | string = {}) {
    const o: HarnessOptions = typeof options === "string" ? { model: options } : options;
    const memory = o.memory === true ? new InMemoryMemory() : typeof o.memory === "string" ? new FileMemory(o.memory) : o.memory || undefined;
    this.agent =
      o.agent ??
      new Agent({
        name: o.name ?? "assistant",
        instructions: o.instructions ?? DEFAULT_INSTRUCTIONS,
        model: o.model,
        tools: o.tools ?? DEFAULT_TOOLS,
        mcpServers: o.mcpServers ?? [],
        memory,
        modelSettings: o.modelSettings ?? {},
      });
    this.session = !o.session || o.session === "memory" ? new InMemorySession() : typeof o.session === "string" ? new FileSession(o.session) : o.session;
    this.limits = o.limits;
    this.trace = o.trace;
    this.approve = o.approve;
  }

  /** @internal options shared by every run */
  runOptions(extra: RunOptions = {}): RunOptions {
    this.token = new CancelToken();
    return { session: this.session, limits: this.limits, trace: this.trace, approve: this.approve, cancelToken: this.token, ...extra };
  }

  async run(prompt: string, options: RunOptions = {}): Promise<RunResult<any>> {
    const result = await run(this.agent, prompt, this.runOptions(options));
    this.usage.add(result.usage);
    return result;
  }

  /** Send a message and return the answer (text, or structured output). */
  async ask(prompt: string, options: RunOptions = {}): Promise<any> {
    return (await this.run(prompt, options)).output;
  }

  /** Yield text as it is generated. */
  async *stream(prompt: string, options: RunOptions = {}): AsyncIterable<string> {
    const s = runStream(this.agent, prompt, this.runOptions(options));
    for await (const ev of s) if (ev.type === "text_delta") yield ev.data.delta;
    this.usage.add((await s.result()).usage);
  }

  cancel(): void {
    this.token?.cancel("cancelled by user");
  }

  reset(): Promise<void> {
    return this.session.clear();
  }

  async close(): Promise<void> {
    for (const server of this.agent.mcpServers) await server.close();
  }

  /** Interactive terminal chat. Commands: /exit /reset /usage /tools /help. */
  async chat(options: { input?: NodeJS.ReadableStream; output?: NodeJS.WritableStream; showTools?: boolean } = {}): Promise<void> {
    const { terminalChat } = await import("./chat.js");
    await terminalChat(this, options);
  }
}
