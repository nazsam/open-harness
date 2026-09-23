/** The `Model` interface every provider adapter implements, plus shared HTTP plumbing. */

import { ModelError } from "../errors.js";
import type { Message, ModelRequest, ModelResponse } from "../types.js";

export type ModelEvent =
  | { type: "text_delta"; delta: string }
  | { type: "tool_call"; id: string; name: string; arguments: Record<string, unknown> }
  | { type: "done"; response: ModelResponse };

export interface Model {
  readonly provider: string;
  readonly model: string;
  generate(request: ModelRequest): Promise<ModelResponse>;
  stream(request: ModelRequest): AsyncIterable<ModelEvent>;
}

/** Streaming fallback for models that only implement `generate`. */
export async function* streamFromGenerate(model: Model, request: ModelRequest): AsyncIterable<ModelEvent> {
  const response = await model.generate(request);
  if (response.message.content) yield { type: "text_delta", delta: response.message.content };
  for (const tc of response.message.toolCalls ?? []) yield { type: "tool_call", ...tc };
  yield { type: "done", response };
}

/** Provider payload saved on a message if it came from this provider. */
export function ownRaw(provider: string, message: Message): unknown {
  return message.raw && message.raw.provider === provider ? message.raw.data : undefined;
}

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

const RETRY_STATUS = new Set([408, 409, 429, 500, 502, 503, 504, 529]);

export interface HTTPOptions {
  fetch?: FetchLike;
  maxRetries?: number;
  timeoutSeconds?: number;
}

export class HTTPClient {
  private readonly fetchImpl: FetchLike;
  private readonly maxRetries: number;
  private readonly timeoutMs: number;

  constructor(
    public readonly baseUrl: string,
    public readonly headers: Record<string, string>,
    public readonly provider: string,
    options: HTTPOptions = {},
  ) {
    this.fetchImpl = options.fetch ?? ((input, init) => fetch(input, init));
    this.maxRetries = options.maxRetries ?? 3;
    this.timeoutMs = (options.timeoutSeconds ?? 300) * 1000;
  }

  private url(path: string): string {
    return path.startsWith("http") ? path : `${this.baseUrl.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
  }

  private async backoff(attempt: number, resp?: Response): Promise<void> {
    let delay = Math.min(2 ** attempt, 30) * 1000 + Math.random() * 1000;
    const ra = resp?.headers.get("retry-after");
    if (ra && !Number.isNaN(Number(ra))) delay = Math.min(Number(ra), 60) * 1000;
    await new Promise((r) => setTimeout(r, delay));
  }

  private async error(resp: Response): Promise<ModelError> {
    const text = await resp.text();
    let body: unknown = text;
    let msg: unknown = text;
    try {
      body = JSON.parse(text);
      const b = body as any;
      const err = Array.isArray(b) ? b[0]?.error : (b?.error ?? b);
      msg = typeof err === "object" && err?.message ? err.message : err;
    } catch {
      /* not JSON */
    }
    return new ModelError(`${this.provider} API error ${resp.status}: ${typeof msg === "string" ? msg : JSON.stringify(msg)}`, resp.status, body, this.provider);
  }

  private async send(path: string, body: unknown, extra: Record<string, string>, signal?: AbortSignal): Promise<Response> {
    for (let attempt = 0; ; attempt++) {
      const timeout = AbortSignal.timeout(this.timeoutMs);
      const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;
      let resp: Response;
      try {
        resp = await this.fetchImpl(this.url(path), {
          method: "POST",
          headers: { ...this.headers, ...extra },
          body: JSON.stringify(body),
          signal: combined,
        });
      } catch (e) {
        if (signal?.aborted) throw e;
        if (attempt >= this.maxRetries) throw new ModelError(`${this.provider}: network error: ${(e as Error).message}`, undefined, undefined, this.provider);
        await this.backoff(attempt);
        continue;
      }
      if (resp.status >= 400) {
        if (RETRY_STATUS.has(resp.status) && attempt < this.maxRetries) {
          await resp.body?.cancel();
          await this.backoff(attempt, resp);
          continue;
        }
        throw await this.error(resp);
      }
      return resp;
    }
  }

  async postJSON(path: string, body: unknown, signal?: AbortSignal, headers: Record<string, string> = {}): Promise<any> {
    const resp = await this.send(path, body, headers, signal);
    return resp.json();
  }

  /** POST and yield `{event, data}` pairs from a Server-Sent Events response. */
  async *postSSE(path: string, body: unknown, signal?: AbortSignal, headers: Record<string, string> = {}): AsyncIterable<{ event?: string; data: string }> {
    const resp = await this.send(path, body, { Accept: "text/event-stream", ...headers }, signal);
    if (!resp.body) return;
    yield* parseSSE(resp.body);
  }
}

export async function* readLines(body: ReadableStream<Uint8Array>): AsyncIterable<string> {
  const decoder = new TextDecoder();
  let buffer = "";
  const reader = body.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        yield buffer.slice(0, idx).replace(/\r$/, "");
        buffer = buffer.slice(idx + 1);
      }
    }
    buffer += decoder.decode();
    if (buffer) yield buffer.replace(/\r$/, "");
  } finally {
    reader.releaseLock();
  }
}

export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncIterable<{ event?: string; data: string }> {
  let event: string | undefined;
  let data: string[] = [];
  for await (const line of readLines(body)) {
    if (line === "") {
      if (data.length) yield { event, data: data.join("\n") };
      event = undefined;
      data = [];
      continue;
    }
    if (line.startsWith(":")) continue;
    const i = line.indexOf(":");
    const key = i === -1 ? line : line.slice(0, i);
    let value = i === -1 ? "" : line.slice(i + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (key === "event") event = value;
    else if (key === "data") data.push(value);
  }
  if (data.length) yield { event, data: data.join("\n") };
}

/** Parse tool-call arguments that arrive as a JSON string. */
export function parseArgs(raw: unknown): Record<string, unknown> {
  if (raw === undefined || raw === null || raw === "") return {};
  if (typeof raw === "object") return raw as Record<string, unknown>;
  try {
    const v = JSON.parse(String(raw));
    return v && typeof v === "object" && !Array.isArray(v) ? v : { value: v };
  } catch {
    return { __invalid_json__: raw };
  }
}

export function env(name: string): string | undefined {
  const v = process.env[name];
  return v && v.length ? v : undefined;
}
