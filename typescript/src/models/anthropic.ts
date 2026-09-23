/** Anthropic Messages API adapter (Claude models). */

import { ConfigError, ModelError } from "../errors.js";
import { Message, Usage, type ModelRequest, type ModelResponse, type ToolCall } from "../types.js";
import { env, HTTPClient, ownRaw, parseArgs, type HTTPOptions, type Model, type ModelEvent } from "./base.js";

export const ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1";
export const ANTHROPIC_VERSION = "2023-06-01";
export const DEFAULT_MAX_TOKENS = 16000;

type Block = Record<string, any>;

export interface AnthropicModelOptions extends HTTPOptions {
  apiKey?: string;
  baseUrl?: string;
  headers?: Record<string, string>;
}

export class AnthropicModel implements Model {
  readonly provider = "anthropic";
  readonly http: HTTPClient;

  constructor(
    readonly model: string,
    options: AnthropicModelOptions = {},
  ) {
    const key = options.apiKey ?? env("ANTHROPIC_API_KEY");
    if (!key) throw new ConfigError("anthropic: set ANTHROPIC_API_KEY or pass apiKey");
    const baseUrl = options.baseUrl ?? env("ANTHROPIC_BASE_URL") ?? ANTHROPIC_BASE_URL;
    this.http = new HTTPClient(
      baseUrl,
      { "x-api-key": key, "anthropic-version": ANTHROPIC_VERSION, "Content-Type": "application/json", ...(options.headers ?? {}) },
      "anthropic",
      options,
    );
  }

  buildBody(req: ModelRequest, stream: boolean): Record<string, unknown> {
    const s = req.settings ?? {};
    const body: Record<string, unknown> = {
      model: this.model,
      max_tokens: s.maxOutputTokens ?? DEFAULT_MAX_TOKENS,
      messages: this.messages(req.messages),
    };
    const system = [req.system, ...req.messages.filter((m) => m.role === "system").map((m) => m.content)].filter(Boolean);
    if (system.length) body.system = system.join("\n\n");
    if (req.tools?.length) {
      body.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
      if (s.toolChoice) {
        const map: Record<string, Block> = { auto: { type: "auto" }, required: { type: "any" }, none: { type: "none" } };
        const tc: Block = map[s.toolChoice] ?? { type: "tool", name: s.toolChoice };
        if (s.parallelToolCalls === false && tc.type !== "none") tc.disable_parallel_tool_use = true;
        body.tool_choice = tc;
      } else if (s.parallelToolCalls === false) {
        body.tool_choice = { type: "auto", disable_parallel_tool_use: true };
      }
    }
    if (s.temperature !== undefined) body.temperature = s.temperature;
    if (s.topP !== undefined) body.top_p = s.topP;
    if (req.outputSchema) body.output_config = { format: { type: "json_schema", schema: req.outputSchema.schema } };
    if (stream) body.stream = true;
    Object.assign(body, s.extra ?? {});
    return body;
  }

  private messages(messages: Message[]): { role: string; content: Block[] }[] {
    const out: { role: string; content: Block[] }[] = [];
    const push = (role: string, blocks: Block[]) => {
      if (!blocks.length) return;
      const last = out[out.length - 1];
      if (last && last.role === role) last.content.push(...blocks);
      else out.push({ role, content: [...blocks] });
    };
    for (const m of messages) {
      if (m.role === "system") continue;
      if (m.role === "user") push("user", [{ type: "text", text: m.content || "(empty)" }]);
      else if (m.role === "tool") {
        const block: Block = { type: "tool_result", tool_use_id: m.toolCallId, content: m.content || "(empty)" };
        if (m.isError) block.is_error = true;
        push("user", [block]);
      } else {
        const raw = ownRaw(this.provider, m) as Block[] | undefined;
        if (raw) {
          push("assistant", raw.map((b) => ({ ...b })));
          continue;
        }
        const blocks: Block[] = [];
        if (m.content) blocks.push({ type: "text", text: m.content });
        for (const tc of m.toolCalls ?? []) blocks.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.arguments });
        push("assistant", blocks.length ? blocks : [{ type: "text", text: "(empty)" }]);
      }
    }
    // The Messages API requires tool_result blocks to come first in a user turn.
    for (const msg of out) {
      if (msg.role === "user") msg.content.sort((a, b) => (a.type === "tool_result" ? 0 : 1) - (b.type === "tool_result" ? 0 : 1));
    }
    return out;
  }

  private toMessage(blocks: Block[]): Message {
    const text = blocks.filter((b) => b.type === "text").map((b) => b.text ?? "").join("");
    const calls: ToolCall[] = blocks.filter((b) => b.type === "tool_use").map((b) => ({ id: b.id, name: b.name, arguments: b.input ?? {} }));
    const msg = Message.assistant(text, calls);
    msg.raw = { provider: this.provider, data: blocks };
    return msg;
  }

  private usage(u: any): Usage {
    const input = (u?.input_tokens ?? 0) + (u?.cache_read_input_tokens ?? 0) + (u?.cache_creation_input_tokens ?? 0);
    return new Usage({ inputTokens: input, outputTokens: u?.output_tokens ?? 0, requests: 1 });
  }

  async generate(req: ModelRequest): Promise<ModelResponse> {
    const data = await this.http.postJSON("messages", this.buildBody(req, false), req.signal);
    if (data.type === "error") throw new ModelError(`anthropic: ${JSON.stringify(data.error)}`, undefined, data, this.provider);
    return { message: this.toMessage(data.content ?? []), usage: this.usage(data.usage), stopReason: data.stop_reason };
  }

  async *stream(req: ModelRequest): AsyncIterable<ModelEvent> {
    const blocks = new Map<number, Block>();
    const partialJson = new Map<number, string>();
    let usage = new Usage({ requests: 1 });
    let stop: string | undefined;
    for await (const { event, data } of this.http.postSSE("messages", this.buildBody(req, true), req.signal)) {
      const p = JSON.parse(data);
      const type = p.type ?? event;
      if (type === "message_start") usage = this.usage(p.message?.usage);
      else if (type === "content_block_start") {
        blocks.set(p.index, { ...p.content_block });
        if (p.content_block?.type === "tool_use") partialJson.set(p.index, "");
      } else if (type === "content_block_delta") {
        const b = blocks.get(p.index) ?? { type: "text", text: "" };
        blocks.set(p.index, b);
        const d = p.delta ?? {};
        if (d.type === "text_delta") {
          b.text = (b.text ?? "") + d.text;
          yield { type: "text_delta", delta: d.text };
        } else if (d.type === "input_json_delta") partialJson.set(p.index, (partialJson.get(p.index) ?? "") + (d.partial_json ?? ""));
        else if (d.type === "thinking_delta") b.thinking = (b.thinking ?? "") + (d.thinking ?? "");
        else if (d.type === "signature_delta") b.signature = (b.signature ?? "") + (d.signature ?? "");
        else if (d.type === "citations_delta") (b.citations ??= []).push(d.citation);
      } else if (type === "content_block_stop") {
        if (partialJson.has(p.index)) {
          const b = blocks.get(p.index)!;
          b.input = parseArgs(partialJson.get(p.index));
          partialJson.delete(p.index);
          yield { type: "tool_call", id: b.id, name: b.name, arguments: b.input };
        }
      } else if (type === "message_delta") {
        stop = p.delta?.stop_reason ?? stop;
        if (p.usage?.output_tokens !== undefined) usage.outputTokens = p.usage.output_tokens;
      } else if (type === "error") {
        throw new ModelError(`anthropic stream error: ${JSON.stringify(p.error)}`, undefined, p, this.provider);
      }
    }
    const ordered = [...blocks.entries()].sort(([a], [b]) => a - b).map(([, b]) => b);
    yield { type: "done", response: { message: this.toMessage(ordered), usage, stopReason: stop } };
  }
}
