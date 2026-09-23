/**
 * Tracing: see every model call, tool call, handoff and guardrail in a run.
 *
 * Each run creates a `Trace` of nested `Span`s. Finished traces go to
 * processors: print to the console, append JSON Lines, or forward anywhere.
 *
 *   run(agent, "hi", { trace: new ConsoleProcessor() })
 *   addTraceProcessor(new JSONLProcessor("traces.jsonl"))
 *   // or with no code change: OPENHARNESS_TRACE=console | OPENHARNESS_TRACE=./traces.jsonl
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { newId } from "./types.js";

export class Span {
  readonly id = newId("span");
  readonly start = Date.now() / 1000;
  end?: number;
  error?: string;
  constructor(
    readonly kind: string,
    readonly name: string,
    readonly traceId: string,
    readonly parentId: string | undefined,
    readonly attributes: Record<string, unknown> = {},
  ) {}

  get durationMs(): number | undefined {
    return this.end === undefined ? undefined : Math.round((this.end - this.start) * 10000) / 10;
  }

  set(attrs: Record<string, unknown>): void {
    Object.assign(this.attributes, attrs);
  }

  toJSON() {
    return {
      id: this.id,
      trace_id: this.traceId,
      parent_id: this.parentId ?? null,
      kind: this.kind,
      name: this.name,
      start: this.start,
      end: this.end ?? null,
      duration_ms: this.durationMs ?? null,
      attributes: this.attributes,
      error: this.error ?? null,
    };
  }
}

export interface TraceProcessor {
  onSpanStart?(span: Span): void;
  onSpanEnd?(span: Span): void;
  onTraceEnd?(trace: Trace): void;
}

export class Trace {
  readonly id = newId("trace");
  readonly spans: Span[] = [];

  constructor(
    readonly name: string,
    readonly processors: TraceProcessor[] = [],
    readonly metadata: Record<string, unknown> = {},
  ) {}

  startSpan(kind: string, name: string, parent?: Span, attributes: Record<string, unknown> = {}): Span {
    const span = new Span(kind, name, this.id, parent?.id, { ...attributes });
    this.spans.push(span);
    this.emit((p) => p.onSpanStart?.(span));
    return span;
  }

  endSpan(span: Span, error?: unknown): void {
    if (span.end !== undefined) return;
    span.end = Date.now() / 1000;
    if (error) span.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    this.emit((p) => p.onSpanEnd?.(span));
  }

  /** Run `fn` inside a span, ending it (and recording any error) afterwards. */
  async span<T>(kind: string, name: string, parent: Span | undefined, attrs: Record<string, unknown>, fn: (span: Span) => Promise<T>): Promise<T> {
    const span = this.startSpan(kind, name, parent, attrs);
    try {
      const out = await fn(span);
      this.endSpan(span);
      return out;
    } catch (e) {
      this.endSpan(span, e);
      throw e;
    }
  }

  finish(): void {
    this.emit((p) => p.onTraceEnd?.(this));
  }

  private emit(fn: (p: TraceProcessor) => void): void {
    for (const p of this.processors) {
      try {
        fn(p);
      } catch (e) {
        console.error(`[openharness] trace processor failed: ${(e as Error).message}`);
      }
    }
  }

  toJSON() {
    return { id: this.id, name: this.name, metadata: this.metadata, spans: this.spans.map((s) => s.toJSON()) };
  }
}

/** Prints one line per finished span, indented by depth. */
export class ConsoleProcessor implements TraceProcessor {
  private depth = new Map<string, number>();
  constructor(private readonly write: (line: string) => void = (l) => process.stderr.write(l + "\n")) {}

  onSpanStart(span: Span): void {
    this.depth.set(span.id, (span.parentId ? (this.depth.get(span.parentId) ?? -1) : -1) + 1);
  }

  onSpanEnd(span: Span): void {
    const pad = "  ".repeat(this.depth.get(span.id) ?? 0);
    let extra = "";
    const u = span.attributes.usage as any;
    if (span.kind === "model" && u) extra = ` in=${u.inputTokens} out=${u.outputTokens}`;
    this.write(`[trace] ${pad}${span.kind}:${span.name} ${span.durationMs}ms${extra}${span.error ? ` ERROR ${span.error}` : ""}`);
  }
}

/** Appends each finished trace as one JSON line. */
export class JSONLProcessor implements TraceProcessor {
  constructor(readonly path: string) {}
  onTraceEnd(trace: Trace): void {
    mkdirSync(dirname(this.path), { recursive: true });
    appendFileSync(this.path, JSON.stringify(trace) + "\n", "utf8");
  }
}

/** Keeps finished traces in memory. Handy for tests and dashboards. */
export class MemoryProcessor implements TraceProcessor {
  readonly traces: Trace[] = [];
  onTraceEnd(trace: Trace): void {
    this.traces.push(trace);
  }
}

const GLOBAL: TraceProcessor[] = [];
export const addTraceProcessor = (p: TraceProcessor): void => void GLOBAL.push(p);
export const clearTraceProcessors = (): void => void GLOBAL.splice(0);

export function processorsFromEnv(): TraceProcessor[] {
  const v = (process.env.OPENHARNESS_TRACE ?? "").trim();
  if (!v || ["0", "false", "off", "none"].includes(v.toLowerCase())) return [];
  if (["1", "true", "console", "stderr"].includes(v.toLowerCase())) return [new ConsoleProcessor()];
  return [new JSONLProcessor(v)];
}

export function globalProcessors(): TraceProcessor[] {
  return [...GLOBAL, ...processorsFromEnv()];
}
