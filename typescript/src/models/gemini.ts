/** Google Gemini adapter (Gemini API `generateContent`). */

import { ConfigError, ModelError } from "../errors.js";
import { Message, newId, Usage, type ModelRequest, type ModelResponse, type ToolCall } from "../types.js";
import { env, HTTPClient, ownRaw, type HTTPOptions, type Model, type ModelEvent } from "./base.js";

export const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

type Part = Record<string, any>;

export interface GeminiModelOptions extends HTTPOptions {
  apiKey?: string;
  baseUrl?: string;
  headers?: Record<string, string>;
}

export class GeminiModel implements Model {
  readonly provider = "gemini";
  readonly model: string;
  readonly http: HTTPClient;

  constructor(model: string, options: GeminiModelOptions = {}) {
    this.model = model.replace(/^models\//, "");
    const key = options.apiKey ?? env("GEMINI_API_KEY") ?? env("GOOGLE_API_KEY");
    if (!key) throw new ConfigError("gemini: set GEMINI_API_KEY or pass apiKey");
    const baseUrl = options.baseUrl ?? env("GEMINI_BASE_URL") ?? GEMINI_BASE_URL;
    this.http = new HTTPClient(baseUrl, { "x-goog-api-key": key, "Content-Type": "application/json", ...(options.headers ?? {}) }, "gemini", options);
  }

  buildBody(req: ModelRequest): Record<string, unknown> {
    const s = req.settings ?? {};
    const body: Record<string, unknown> = { contents: this.contents(req.messages) };
    const system = [req.system, ...req.messages.filter((m) => m.role === "system").map((m) => m.content)].filter(Boolean);
    if (system.length) body.systemInstruction = { parts: [{ text: system.join("\n\n") }] };
    if (req.tools?.length) {
      body.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parametersJsonSchema: t.parameters })) }];
      if (s.toolChoice) {
        const mode = ({ auto: "AUTO", required: "ANY", none: "NONE" } as Record<string, string>)[s.toolChoice];
        const cfg: Record<string, unknown> = { mode: mode ?? "ANY" };
        if (!mode) cfg.allowedFunctionNames = [s.toolChoice];
        body.toolConfig = { functionCallingConfig: cfg };
      }
    }
    const gen: Record<string, unknown> = {};
    if (s.temperature !== undefined) gen.temperature = s.temperature;
    if (s.topP !== undefined) gen.topP = s.topP;
    if (s.maxOutputTokens !== undefined) gen.maxOutputTokens = s.maxOutputTokens;
    if (req.outputSchema) {
      gen.responseMimeType = "application/json";
      gen.responseJsonSchema = req.outputSchema.schema;
    }
    if (Object.keys(gen).length) body.generationConfig = gen;
    Object.assign(body, s.extra ?? {});
    return body;
  }

  private contents(messages: Message[]): { role: string; parts: Part[] }[] {
    const out: { role: string; parts: Part[] }[] = [];
    const push = (role: string, parts: Part[]) => {
      const last = out[out.length - 1];
      if (last && last.role === role) last.parts.push(...parts);
      else out.push({ role, parts: [...parts] });
    };
    for (const m of messages) {
      if (m.role === "system") continue;
      if (m.role === "user") push("user", [{ text: m.content || " " }]);
      else if (m.role === "tool") {
        let result: unknown = m.content;
        try {
          result = JSON.parse(m.content);
        } catch {
          /* keep string */
        }
        const fr: Part = { name: m.name ?? "tool", response: m.isError ? { error: result } : { result } };
        if (m.toolCallId && !m.toolCallId.startsWith("gcall_")) fr.id = m.toolCallId;
        push("user", [{ functionResponse: fr }]);
      } else {
        const raw = ownRaw(this.provider, m) as Part[] | undefined;
        if (raw) {
          push("model", raw.map((p) => ({ ...p })));
          continue;
        }
        const parts: Part[] = [];
        if (m.content) parts.push({ text: m.content });
        for (const tc of m.toolCalls ?? []) {
          const fc: Part = { name: tc.name, args: tc.arguments };
          if (!tc.id.startsWith("gcall_")) fc.id = tc.id;
          parts.push({ functionCall: fc });
        }
        push("model", parts.length ? parts : [{ text: " " }]);
      }
    }
    return out;
  }

  private usage(u: any): Usage {
    return new Usage({ inputTokens: u?.promptTokenCount ?? 0, outputTokens: (u?.candidatesTokenCount ?? 0) + (u?.thoughtsTokenCount ?? 0), requests: 1 });
  }

  private toMessage(parts: Part[]): Message {
    const text = parts.filter((p) => "text" in p && !p.thought).map((p) => p.text).join("");
    const calls: ToolCall[] = parts
      .filter((p) => p.functionCall)
      .map((p) => ({ id: p.functionCall.id ?? newId("gcall"), name: p.functionCall.name, arguments: p.functionCall.args ?? {} }));
    const msg = Message.assistant(text, calls);
    msg.raw = { provider: this.provider, data: parts };
    return msg;
  }

  private candidates(data: any): any[] {
    const c = data.candidates ?? [];
    if (!c.length && data.promptFeedback?.blockReason)
      throw new ModelError(`gemini: prompt blocked (${data.promptFeedback.blockReason})`, undefined, data, this.provider);
    return c;
  }

  async generate(req: ModelRequest): Promise<ModelResponse> {
    const data = await this.http.postJSON(`models/${this.model}:generateContent`, this.buildBody(req), req.signal);
    const cand = this.candidates(data)[0];
    return { message: this.toMessage(cand?.content?.parts ?? []), usage: this.usage(data.usageMetadata), stopReason: cand?.finishReason };
  }

  async *stream(req: ModelRequest): AsyncIterable<ModelEvent> {
    const parts: Part[] = [];
    let usage = new Usage({ requests: 1 });
    let finish: string | undefined;
    const plain = (p: Part | undefined) => p && "text" in p && !p.thought && !p.thoughtSignature;
    for await (const { data } of this.http.postSSE(`models/${this.model}:streamGenerateContent?alt=sse`, this.buildBody(req), req.signal)) {
      const chunk = JSON.parse(data);
      if (chunk.error) throw new ModelError(`gemini: ${JSON.stringify(chunk.error)}`, undefined, chunk, this.provider);
      if (chunk.usageMetadata) usage = this.usage(chunk.usageMetadata);
      const cand = this.candidates(chunk)[0];
      if (!cand) continue;
      finish = cand.finishReason ?? finish;
      for (const p of cand.content?.parts ?? []) {
        if (plain(p) && plain(parts[parts.length - 1])) parts[parts.length - 1].text += p.text;
        else parts.push({ ...p });
        if ("text" in p && !p.thought && p.text) yield { type: "text_delta", delta: p.text };
      }
    }
    const msg = this.toMessage(parts);
    for (const tc of msg.toolCalls ?? []) yield { type: "tool_call", ...tc };
    yield { type: "done", response: { message: msg, usage, stopReason: finish } };
  }
}
