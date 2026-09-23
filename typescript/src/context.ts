/** Run-scoped state: limits, cancellation, and the context passed to tools. */

import type { Agent } from "./agent.js";
import type { Trace } from "./tracing.js";
import { Usage, type Message } from "./types.js";

/**
 * Hard stops so an agent can never run forever. Every limit is optional.
 * Defaults: 20 model turns, 50 tool calls, 10 minutes.
 */
export interface RunLimits {
  maxTurns?: number | null;
  maxToolCalls?: number | null;
  maxTotalTokens?: number | null;
  timeoutSeconds?: number | null;
  /** Re-asks when structured output fails validation. */
  maxOutputRetries?: number;
}

export const DEFAULT_LIMITS: Required<RunLimits> = {
  maxTurns: 20,
  maxToolCalls: 50,
  maxTotalTokens: null,
  timeoutSeconds: 600,
  maxOutputRetries: 2,
};

/**
 * Cooperative cancellation. Call `cancel()` from anywhere to stop a run.
 * You can also pass a standard `AbortSignal` as `signal` in run options.
 */
export class CancelToken {
  private controller = new AbortController();
  reason?: string;

  cancel(reason = "cancelled"): void {
    this.reason = reason;
    this.controller.abort(reason);
  }

  get cancelled(): boolean {
    return this.controller.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.controller.signal;
  }
}

/**
 * Passed to tools, guardrails, dynamic instructions and hooks. `deps` is
 * whatever you pass to `run(..., { deps })`; it is never sent to the model.
 */
export class RunContext<TDeps = unknown> {
  messages: Message[] = [];
  usage = new Usage();
  turn = 0;
  toolCalls = 0;
  readonly startedAt = Date.now();
  /** Scratch space shared across the run. */
  state: Record<string, unknown> = {};

  constructor(
    public agent: Agent<any>,
    public deps: TDeps,
    public trace: Trace,
    public signal: AbortSignal,
  ) {}

  get elapsedSeconds(): number {
    return (Date.now() - this.startedAt) / 1000;
  }
}
