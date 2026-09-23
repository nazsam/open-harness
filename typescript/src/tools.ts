/**
 * Tools: functions the model can call.
 *
 *   const getWeather = tool({
 *     name: "get_weather",
 *     description: "Get the current weather for a city.",
 *     parameters: s.object({ city: s.string().describe('City name, e.g. "Toronto"') }),
 *     execute: async ({ city }) => `18C in ${city}`,
 *   });
 */

import type { RunContext } from "./context.js";
import { ToolError } from "./errors.js";
import { Schema, SchemaValidationError, toJSONSchema, validate } from "./schema.js";
import type { JSONSchema, ToolSpec } from "./types.js";

export interface Tool {
  name: string;
  description: string;
  parameters: JSONSchema;
  execute: (args: Record<string, any>, ctx: RunContext<any>) => unknown | Promise<unknown>;
  needsApproval?: boolean | ((args: Record<string, any>) => boolean);
  timeoutSeconds?: number;
  metadata?: Record<string, unknown>;
}

export interface ToolOptions<A> {
  name: string;
  description: string;
  parameters?: Schema<A> | JSONSchema;
  execute: (args: A, ctx: RunContext<any>) => unknown | Promise<unknown>;
  needsApproval?: boolean | ((args: A) => boolean);
  timeoutSeconds?: number;
}

const EMPTY: JSONSchema = { type: "object", properties: {}, additionalProperties: false };

/** Define a tool. Argument types are inferred from an `s.object(...)` schema. */
export function tool<A = Record<string, any>>(options: ToolOptions<A>): Tool {
  return {
    name: options.name,
    description: options.description,
    parameters: toJSONSchema(options.parameters) ?? EMPTY,
    execute: options.execute as Tool["execute"],
    needsApproval: options.needsApproval as Tool["needsApproval"],
    timeoutSeconds: options.timeoutSeconds,
    metadata: { source: "function" },
  };
}

export function toolSpec(t: Tool): ToolSpec {
  return { name: t.name, description: t.description, parameters: t.parameters };
}

export function requiresApproval(t: Tool, args: Record<string, unknown>): boolean {
  return typeof t.needsApproval === "function" ? Boolean(t.needsApproval(args)) : Boolean(t.needsApproval);
}

export async function invokeTool(t: Tool, args: Record<string, unknown>, ctx: RunContext<any>): Promise<unknown> {
  try {
    validate(args, t.parameters);
  } catch (e) {
    if (e instanceof SchemaValidationError) throw new ToolError(`Invalid arguments for ${t.name}: ${e.message}`);
    throw e;
  }
  const work = Promise.resolve().then(() => t.execute(args, ctx));
  if (!t.timeoutSeconds) return work;
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new ToolError(`Tool ${t.name} timed out after ${t.timeoutSeconds}s`)), t.timeoutSeconds! * 1000);
  });
  try {
    return await Promise.race([work, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

export { Schema };
