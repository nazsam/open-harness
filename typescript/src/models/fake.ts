/** A scripted model for tests and offline demos. No network, no API key. */

import { Message, newId, Usage, type ModelRequest, type ModelResponse } from "../types.js";
import type { Model, ModelEvent } from "./base.js";

export type Step =
  | string
  | Message
  | { text?: string; json?: unknown; toolCalls?: { id?: string; name: string; arguments?: Record<string, unknown> }[] }
  | ((req: ModelRequest) => string | Message | { text?: string; json?: unknown; toolCalls?: any[] });

/** Script a tool call: `new FakeModel([call("add", { a: 1, b: 2 }), "The answer is 3"])`. */
export function call(name: string, args: Record<string, unknown> = {}): Step {
  return { toolCalls: [{ name, arguments: args }] };
}

/**
 * Replays a script of responses, one per model call. When the script runs
 * out it repeats `defaultReply` (or echoes the last user message). Every
 * request is recorded in `requests` for assertions.
 */
export class FakeModel implements Model {
  readonly provider = "fake";
  readonly requests: ModelRequest[] = [];
  private script: Step[];

  constructor(
    script: Step[] = [],
    readonly options: { model?: string; defaultReply?: string; chunkSize?: number } = {},
  ) {
    this.script = [...script];
  }

  get model(): string {
    return this.options.model ?? "fake-model";
  }

  private next(req: ModelRequest): Message {
    let step: Step;
    if (this.script.length) step = this.script.shift()!;
    else if (this.options.defaultReply !== undefined) step = this.options.defaultReply;
    else {
      const last = [...req.messages].reverse().find((m) => m.role === "user")?.content ?? "";
      step = `echo: ${last}`;
    }
    if (typeof step === "function") step = step(req);
    if (typeof step === "string") return Message.assistant(step);
    if ("role" in step) return step as Message;
    const s = step as { text?: string; json?: unknown; toolCalls?: any[] };
    const calls = (s.toolCalls ?? []).map((tc) => ({ id: tc.id ?? newId("call"), name: tc.name, arguments: tc.arguments ?? {} }));
    return Message.assistant(s.json !== undefined ? JSON.stringify(s.json) : (s.text ?? ""), calls);
  }

  async generate(request: ModelRequest): Promise<ModelResponse> {
    this.requests.push(request);
    const message = this.next(request);
    const tokens = Math.max(1, Math.floor(message.content.length / 4));
    return {
      message,
      usage: new Usage({ inputTokens: 10, outputTokens: tokens, requests: 1 }),
      stopReason: message.toolCalls?.length ? "tool_use" : "stop",
    };
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelEvent> {
    const response = await this.generate(request);
    const size = this.options.chunkSize ?? 4;
    const text = response.message.content;
    for (let i = 0; i < text.length; i += size) yield { type: "text_delta", delta: text.slice(i, i + size) };
    for (const tc of response.message.toolCalls ?? []) yield { type: "tool_call", ...tc };
    yield { type: "done", response };
  }
}
