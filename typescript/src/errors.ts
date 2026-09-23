/** Exception hierarchy. Every SDK error extends `OpenHarnessError`. */

import type { RunResult } from "./result.js";

export class OpenHarnessError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** The model provider returned an error or an unusable response. */
export class ModelError extends OpenHarnessError {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly body?: unknown,
    public readonly provider?: string,
  ) {
    super(message);
  }
}

/** Invalid configuration, such as an unknown provider or a missing API key. */
export class ConfigError extends OpenHarnessError {}

/** Throw inside a tool to send a clean error message back to the model. */
export class ToolError extends OpenHarnessError {}

/** Base class for errors that end a run early. `partial` holds what was done so far. */
export class RunStopped extends OpenHarnessError {
  partial?: RunResult;
  constructor(message: string, partial?: RunResult) {
    super(message);
    this.partial = partial;
  }
}

export class MaxTurnsExceeded extends RunStopped {}
export class MaxToolCallsExceeded extends RunStopped {}
export class TokenBudgetExceeded extends RunStopped {}
export class RunTimeout extends RunStopped {}
export class RunCancelled extends RunStopped {}
/** The model could not produce output matching the agent's output type after retries. */
export class OutputValidationError extends RunStopped {}

export class GuardrailTripped extends RunStopped {
  constructor(
    message: string,
    public readonly guardrail: string,
    public readonly stage: "input" | "output",
    public readonly info?: unknown,
  ) {
    super(message);
  }
}

export class MCPError extends OpenHarnessError {
  constructor(
    message: string,
    public readonly code?: number,
    public readonly data?: unknown,
  ) {
    super(message);
  }
}
