/** The `Agent`: instructions + model + tools, and how agents compose. */

import type { RunContext, RunLimits } from "./context.js";
import type { Guardrail } from "./guardrails.js";
import type { MCPServer } from "./mcp.js";
import type { Memory } from "./memory.js";
import { getModel, type Model } from "./models/index.js";
import { Schema, toJSONSchema } from "./schema.js";
import type { Tool } from "./tools.js";
import type { JSONSchema, Message, ModelSettings } from "./types.js";

export type Instructions = string | ((ctx: RunContext<any>) => string | Promise<string>);

const slug = (name: string) =>
  name
    .replace(/[^a-zA-Z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase() || "agent";

/**
 * Lets one agent pass the conversation to another. The model sees a tool
 * named `transfer_to_<agent>`; when it calls it, the target agent takes over.
 */
export class Handoff {
  constructor(
    readonly agent: Agent<any>,
    readonly options: { toolName?: string; description?: string; inputFilter?: (history: Message[]) => Message[] } = {},
  ) {}

  get name(): string {
    return this.options.toolName ?? `transfer_to_${slug(this.agent.name)}`;
  }

  tool(): Tool {
    return {
      name: this.name,
      description:
        this.options.description ?? `Hand the conversation to the ${this.agent.name} agent.${this.agent.description ? ` ${this.agent.description}` : ""}`,
      parameters: { type: "object", properties: {}, additionalProperties: false },
      execute: () => `Transferred to ${this.agent.name}.`,
      metadata: { handoff: this },
    };
  }
}

export interface AgentOptions<TOutput> {
  /** Shown in traces and used for handoff tool names. */
  name?: string;
  /** System prompt, or a function of the run context that returns one. */
  instructions?: Instructions;
  /** `"provider:model"` string or a `Model`. Defaults to the environment. */
  model?: string | Model;
  tools?: Tool[];
  /** Agents (or `Handoff`s) this agent may pass control to. */
  handoffs?: (Agent<any> | Handoff)[];
  /** Structured output: an `s.object(...)` schema or a raw JSON Schema. */
  outputType?: Schema<TOutput> | JSONSchema;
  inputGuardrails?: Guardrail[];
  outputGuardrails?: Guardrail[];
  mcpServers?: MCPServer[];
  /** Long-term memory store. Adds `remember`/`recall` tools. */
  memory?: Memory;
  modelSettings?: ModelSettings;
  /** Default limits when this agent starts a run. */
  limits?: RunLimits;
  /** One line used when this agent is offered as a handoff or tool. */
  description?: string;
}

export class Agent<TOutput = string> {
  name: string;
  instructions: Instructions;
  model?: string | Model;
  tools: Tool[];
  handoffs: (Agent<any> | Handoff)[];
  outputType?: JSONSchema;
  inputGuardrails: Guardrail[];
  outputGuardrails: Guardrail[];
  mcpServers: MCPServer[];
  memory?: Memory;
  modelSettings: ModelSettings;
  limits?: RunLimits;
  description: string;
  private resolved?: Model;

  constructor(options: AgentOptions<TOutput> = {}) {
    this.name = options.name ?? "assistant";
    this.instructions = options.instructions ?? "You are a helpful assistant.";
    this.model = options.model;
    this.tools = options.tools ?? [];
    this.handoffs = options.handoffs ?? [];
    this.outputType = toJSONSchema(options.outputType);
    this.inputGuardrails = options.inputGuardrails ?? [];
    this.outputGuardrails = options.outputGuardrails ?? [];
    this.mcpServers = options.mcpServers ?? [];
    this.memory = options.memory;
    this.modelSettings = options.modelSettings ?? {};
    this.limits = options.limits;
    this.description = options.description ?? "";
  }

  getModel(override?: string | Model): Model {
    if (override) return getModel(override);
    this.resolved ??= getModel(this.model);
    return this.resolved;
  }

  async renderInstructions(ctx: RunContext<any>): Promise<string> {
    return typeof this.instructions === "function" ? String(await this.instructions(ctx)) : this.instructions;
  }

  handoffObjects(): Handoff[] {
    return this.handoffs.map((h) => (h instanceof Handoff ? h : new Handoff(h)));
  }

  /** Copy the agent with some options changed. */
  clone<T = TOutput>(changes: AgentOptions<T> = {}): Agent<T> {
    return new Agent<T>({
      name: this.name,
      instructions: this.instructions,
      model: this.model,
      tools: this.tools,
      handoffs: this.handoffs,
      outputType: this.outputType as JSONSchema,
      inputGuardrails: this.inputGuardrails,
      outputGuardrails: this.outputGuardrails,
      mcpServers: this.mcpServers,
      memory: this.memory,
      modelSettings: this.modelSettings,
      limits: this.limits,
      description: this.description,
      ...changes,
    });
  }

  /**
   * Expose this agent as a tool another agent can call (agents as workers).
   * Unlike a handoff, control returns to the calling agent with the result.
   */
  asTool(options: { name?: string; description?: string; outputExtractor?: (output: TOutput) => unknown } = {}): Tool {
    return {
      name: options.name ?? slug(this.name),
      description: options.description ?? (this.description || `Ask the ${this.name} agent. Returns its answer.`),
      parameters: {
        type: "object",
        properties: { input: { type: "string", description: "The full task or question for this agent." } },
        required: ["input"],
        additionalProperties: false,
      },
      execute: async (args, ctx) => {
        const { run } = await import("./runner.js");
        const result = await run(this, String(args.input), { deps: ctx.deps, signal: ctx.signal, parentTrace: ctx.trace });
        ctx.usage.add(result.usage);
        return options.outputExtractor ? options.outputExtractor(result.output) : result.output;
      },
      metadata: { agent: this.name },
    };
  }
}
