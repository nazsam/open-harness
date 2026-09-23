/**
 * Core data types shared by every part of the SDK.
 *
 * Messages use a provider-neutral shape so a conversation can continue on a
 * different provider. Adapters may attach their original payload in `raw` so
 * provider-specific details (such as reasoning signatures) survive a round
 * trip on the same provider.
 */

export type Role = "system" | "user" | "assistant" | "tool";
export type JSONSchema = Record<string, unknown>;

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface Message {
  role: Role;
  content: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
  /** Tool name for tool results, agent name for assistant messages. */
  name?: string;
  isError?: boolean;
  raw?: { provider: string; data: unknown };
}

export const Message = {
  user: (content: string): Message => ({ role: "user", content }),
  system: (content: string): Message => ({ role: "system", content }),
  assistant: (content = "", toolCalls: ToolCall[] = []): Message => ({ role: "assistant", content, toolCalls }),
  tool: (toolCallId: string, name: string, content: string, isError = false): Message => ({
    role: "tool",
    content,
    toolCallId,
    name,
    isError,
  }),
};

export class Usage {
  inputTokens = 0;
  outputTokens = 0;
  requests = 0;

  constructor(init: Partial<Pick<Usage, "inputTokens" | "outputTokens" | "requests">> = {}) {
    Object.assign(this, init);
  }

  get totalTokens(): number {
    return this.inputTokens + this.outputTokens;
  }

  add(other: Usage): void {
    this.inputTokens += other.inputTokens;
    this.outputTokens += other.outputTokens;
    this.requests += other.requests;
  }

  toJSON() {
    return {
      inputTokens: this.inputTokens,
      outputTokens: this.outputTokens,
      requests: this.requests,
      totalTokens: this.totalTokens,
    };
  }
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: JSONSchema;
}

export interface OutputSchema {
  name: string;
  schema: JSONSchema;
  strict?: boolean;
}

export interface ModelSettings {
  temperature?: number;
  topP?: number;
  maxOutputTokens?: number;
  /** "auto" | "required" | "none" | a tool name */
  toolChoice?: string;
  parallelToolCalls?: boolean;
  /** Merged into the provider request body as-is. */
  extra?: Record<string, unknown>;
}

export function mergeSettings(a: ModelSettings = {}, b: ModelSettings = {}): ModelSettings {
  const out: ModelSettings = { ...a };
  for (const [k, v] of Object.entries(b)) {
    if (v !== undefined && k !== "extra") (out as Record<string, unknown>)[k] = v;
  }
  out.extra = { ...(a.extra ?? {}), ...(b.extra ?? {}) };
  return out;
}

export interface ModelRequest {
  system?: string;
  messages: Message[];
  tools?: ToolSpec[];
  outputSchema?: OutputSchema;
  settings?: ModelSettings;
  signal?: AbortSignal;
}

export interface ModelResponse {
  message: Message;
  usage: Usage;
  stopReason?: string;
}

/**
 * Events yielded while a run streams:
 * agent_start {agent} | text_delta {delta} | tool_call {id,name,arguments} |
 * tool_result {id,name,output,isError} | handoff {from,to} | message {message} | run_end {result}
 */
export interface StreamEvent {
  type: "agent_start" | "text_delta" | "tool_call" | "tool_result" | "handoff" | "message" | "run_end";
  data: Record<string, any>;
  agent?: string;
  timestamp: number;
}

let counter = 0;
export function newId(prefix = "id"): string {
  counter = (counter + 1) % 1e6;
  const rand = Math.random().toString(36).slice(2, 12);
  return `${prefix}_${Date.now().toString(36)}${counter.toString(36)}${rand}`;
}

export function toJSONText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === undefined) return "";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** Serialise a message in the same JSON shape the Python SDK uses (snake_case keys). */
export function messageToJSON(m: Message): Record<string, unknown> {
  const d: Record<string, unknown> = { role: m.role, content: m.content };
  if (m.toolCalls?.length) d.tool_calls = m.toolCalls;
  if (m.toolCallId !== undefined) d.tool_call_id = m.toolCallId;
  if (m.name !== undefined) d.name = m.name;
  if (m.isError) d.is_error = true;
  if (m.raw !== undefined) d.raw = m.raw;
  return d;
}

export function messageFromJSON(d: Record<string, any>): Message {
  const m: Message = { role: d.role, content: d.content ?? "" };
  const calls = d.tool_calls ?? d.toolCalls;
  if (calls?.length) m.toolCalls = calls.map((c: any) => ({ id: c.id, name: c.name, arguments: c.arguments ?? {} }));
  const tcid = d.tool_call_id ?? d.toolCallId;
  if (tcid !== undefined && tcid !== null) m.toolCallId = tcid;
  if (d.name !== undefined && d.name !== null) m.name = d.name;
  if (d.is_error ?? d.isError) m.isError = true;
  if (d.raw) m.raw = d.raw;
  return m;
}
