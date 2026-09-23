/**
 * open-harness: a self-hosted SDK for building AI agents.
 *
 *   import { Agent, run, tool, s } from "openharness";
 *
 *   const add = tool({
 *     name: "add",
 *     description: "Add two numbers.",
 *     parameters: s.object({ a: s.number(), b: s.number() }),
 *     execute: ({ a, b }) => a + b,
 *   });
 *   const agent = new Agent({ instructions: "Be brief.", model: "anthropic:claude-sonnet-5", tools: [add] });
 *   console.log((await run(agent, "What is 2 + 40?")).output);
 */

export { VERSION } from "./version.js";
export { Agent, Handoff, type AgentOptions, type Instructions } from "./agent.js";
export { run, runStream, type RunOptions, type Approver } from "./runner.js";
export { RunStream, toMessages, type RunResult } from "./result.js";
export { Harness, DEFAULT_INSTRUCTIONS, type HarnessOptions } from "./harness.js";
export { tool, invokeTool, type Tool, type ToolOptions } from "./tools.js";
export { s, Schema, validate, extractJson, SchemaValidationError, type Infer } from "./schema.js";
export { RunContext, CancelToken, DEFAULT_LIMITS, type RunLimits } from "./context.js";
export {
  Message,
  Usage,
  messageToJSON,
  messageFromJSON,
  type ToolCall,
  type ModelSettings,
  type ModelRequest,
  type ModelResponse,
  type StreamEvent,
  type JSONSchema,
} from "./types.js";
export {
  getModel,
  defaultModelString,
  OpenAIModel,
  AnthropicModel,
  GeminiModel,
  FakeModel,
  call,
  DEFAULT_MODELS,
  PROVIDERS,
  COMPATIBLE_PRESETS,
  type Model,
  type ModelEvent,
  type FetchLike,
} from "./models/index.js";
export { InMemorySession, FileSession, type Session } from "./sessions.js";
export { InMemoryMemory, FileMemory, memoryTools, type Memory, type MemoryItem } from "./memory.js";
export {
  inputGuardrail,
  outputGuardrail,
  maxLength,
  blockedPatterns,
  pii,
  llmGuardrail,
  type Guardrail,
  type GuardrailResult,
} from "./guardrails.js";
export {
  Trace,
  Span,
  ConsoleProcessor,
  JSONLProcessor,
  MemoryProcessor,
  addTraceProcessor,
  clearTraceProcessors,
  type TraceProcessor,
} from "./tracing.js";
export { MCPServer, MCPServerStdio, MCPServerHTTP, type MCPServerOptions } from "./mcp.js";
export { calculator, currentTime, fetchUrl, safeEval, DEFAULT_TOOLS, ALL_TOOLS } from "./builtinTools.js";
export * from "./errors.js";
