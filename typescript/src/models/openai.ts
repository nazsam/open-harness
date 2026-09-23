/**
 * OpenAI Chat Completions adapter. Also used for any OpenAI-compatible
 * server: Ollama, vLLM, LM Studio, Groq, Together, OpenRouter, DeepSeek,
 * Mistral and others. Point `baseUrl` at the server.
 */

import { ConfigError, ModelError } from "../errors.js";
import { isStrictCompatible } from "../schema.js";
import { Message, newId, Usage, type ModelRequest, type ModelResponse, type ToolCall } from "../types.js";
import { env, HTTPClient, parseArgs, type HTTPOptions, type Model, type ModelEvent } from "./base.js";

export const OPENAI_BASE_URL = "https://api.openai.com/v1";

export interface OpenAIModelOptions extends HTTPOptions {
  apiKey?: string;
  baseUrl?: string;
  providerName?: string;
  apiKeyEnv?: string;
  requireKey?: boolean;
  headers?: Record<string, string>;
}

export class OpenAIModel implements Model {
  readonly provider: string;
  readonly baseUrl: string;
  readonly http: HTTPClient;
  private readonly official: boolean;

  constructor(
    readonly model: string,
    options: OpenAIModelOptions = {},
  ) {
    this.provider = options.providerName ?? "openai";
    const keyEnv = options.apiKeyEnv ?? "OPENAI_API_KEY";
    const key = options.apiKey ?? env(keyEnv);
    if (!key && options.requireKey !== false) throw new ConfigError(`${this.provider}: set ${keyEnv} or pass apiKey`);
    this.baseUrl = options.baseUrl ?? (this.provider === "openai" ? env("OPENAI_BASE_URL") : undefined) ?? OPENAI_BASE_URL;
    this.official = this.baseUrl.replace(/\/$/, "") === OPENAI_BASE_URL;
    const headers: Record<string, string> = { "Content-Type": "application/json", ...(options.headers ?? {}) };
    if (key) headers.Authorization = `Bearer ${key}`;
    this.http = new HTTPClient(this.baseUrl, headers, this.provider, options);
  }

  buildBody(req: ModelRequest, stream: boolean): Record<string, unknown> {
    const messages: Record<string, unknown>[] = [];
    if (req.system) messages.push({ role: "system", content: req.system });
    for (const m of req.messages) messages.push(this.convert(m));
    const body: Record<string, unknown> = { model: this.model, messages };
    const s = req.settings ?? {};
    if (req.tools?.length) {
      body.tools = req.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.parameters } }));
      if (s.toolChoice)
        body.tool_choice = ["auto", "required", "none"].includes(s.toolChoice) ? s.toolChoice : { type: "function", function: { name: s.toolChoice } };
      if (s.parallelToolCalls !== undefined) body.parallel_tool_calls = s.parallelToolCalls;
    }
    if (s.temperature !== undefined) body.temperature = s.temperature;
    if (s.topP !== undefined) body.top_p = s.topP;
    if (s.maxOutputTokens !== undefined) body[this.official ? "max_completion_tokens" : "max_tokens"] = s.maxOutputTokens;
    if (req.outputSchema) {
      const strict = (req.outputSchema.strict ?? true) && isStrictCompatible(req.outputSchema.schema);
      body.response_format = { type: "json_schema", json_schema: { name: req.outputSchema.name, schema: req.outputSchema.schema, strict } };
    }
    if (stream) {
      body.stream = true;
      body.stream_options = { include_usage: true };
    }
    Object.assign(body, s.extra ?? {});
    return body;
  }

  private convert(m: Message): Record<string, unknown> {
    if (m.role === "system" || m.role === "user") return { role: m.role, content: m.content };
    if (m.role === "tool") return { role: "tool", tool_call_id: m.toolCallId, content: m.content || "(empty)" };
    const out: Record<string, unknown> = { role: "assistant", content: m.content || null };
    if (m.toolCalls?.length)
      out.tool_calls = m.toolCalls.map((tc) => ({ id: tc.id, type: "function", function: { name: tc.name, arguments: JSON.stringify(tc.arguments) } }));
    return out;
  }

  private usage(u: any): Usage {
    return new Usage({ inputTokens: u?.prompt_tokens ?? 0, outputTokens: u?.completion_tokens ?? 0, requests: 1 });
  }

  async generate(req: ModelRequest): Promise<ModelResponse> {
    const data = await this.http.postJSON("chat/completions", this.buildBody(req, false), req.signal);
    const choice = data.choices?.[0];
    if (!choice) throw new ModelError(`${this.provider}: response had no choices`, undefined, data, this.provider);
    const msg = choice.message ?? {};
    const calls: ToolCall[] = (msg.tool_calls ?? []).map((tc: any) => ({
      id: tc.id ?? newId("call"),
      name: tc.function.name,
      arguments: parseArgs(tc.function.arguments),
    }));
    return { message: Message.assistant(msg.refusal ?? msg.content ?? "", calls), usage: this.usage(data.usage), stopReason: choice.finish_reason };
  }

  async *stream(req: ModelRequest): AsyncIterable<ModelEvent> {
    const text: string[] = [];
    const calls = new Map<number, { id?: string; name: string; args: string }>();
    let usage = new Usage({ requests: 1 });
    let finish: string | undefined;
    for await (const { data } of this.http.postSSE("chat/completions", this.buildBody(req, true), req.signal)) {
      if (data.trim() === "[DONE]") break;
      const chunk = JSON.parse(data);
      if (chunk.error) throw new ModelError(`${this.provider}: ${JSON.stringify(chunk.error)}`, undefined, chunk, this.provider);
      if (chunk.usage) usage = this.usage(chunk.usage);
      for (const choice of chunk.choices ?? []) {
        const delta = choice.delta ?? {};
        if (delta.content) {
          text.push(delta.content);
          yield { type: "text_delta", delta: delta.content };
        }
        for (const tc of delta.tool_calls ?? []) {
          const slot = calls.get(tc.index ?? 0) ?? { name: "", args: "" };
          if (tc.id) slot.id = tc.id;
          if (tc.function?.name) slot.name += tc.function.name;
          if (tc.function?.arguments) slot.args += tc.function.arguments;
          calls.set(tc.index ?? 0, slot);
        }
        if (choice.finish_reason) finish = choice.finish_reason;
      }
    }
    const toolCalls: ToolCall[] = [...calls.entries()]
      .sort(([a], [b]) => a - b)
      .map(([, c]) => ({ id: c.id ?? newId("call"), name: c.name, arguments: parseArgs(c.args) }));
    for (const tc of toolCalls) yield { type: "tool_call", ...tc };
    yield { type: "done", response: { message: Message.assistant(text.join(""), toolCalls), usage, stopReason: finish } };
  }
}

/** OpenAI-compatible presets: provider -> [baseUrl, api key env var, key required] */
export const COMPATIBLE_PRESETS: Record<string, [string, string, boolean]> = {
  ollama: ["http://localhost:11434/v1", "OLLAMA_API_KEY", false],
  lmstudio: ["http://localhost:1234/v1", "LMSTUDIO_API_KEY", false],
  vllm: ["http://localhost:8000/v1", "VLLM_API_KEY", false],
  groq: ["https://api.groq.com/openai/v1", "GROQ_API_KEY", true],
  together: ["https://api.together.xyz/v1", "TOGETHER_API_KEY", true],
  openrouter: ["https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", true],
  deepseek: ["https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", true],
  mistral: ["https://api.mistral.ai/v1", "MISTRAL_API_KEY", true],
  xai: ["https://api.x.ai/v1", "XAI_API_KEY", true],
};
