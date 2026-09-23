/**
 * Guardrails check input before the model sees it and output before your app does.
 *
 * A guardrail is a function `(ctx, value) => GuardrailResult | boolean`. Set
 * `tripwire: true` to stop the run with `GuardrailTripped`, or return a
 * `replacement` to rewrite the value (for example to redact PII) and carry on.
 */

import type { Agent } from "./agent.js";
import type { RunContext } from "./context.js";

export interface GuardrailResult {
  tripwire: boolean;
  info?: unknown;
  /** When set (and not tripped) the checked value is replaced. */
  replacement?: unknown;
}

export type GuardFn = (ctx: RunContext<any>, value: any) => GuardrailResult | boolean | void | Promise<GuardrailResult | boolean | void>;

export interface Guardrail {
  name: string;
  stage: "input" | "output";
  check: GuardFn;
}

export async function runGuardrail(g: Guardrail, ctx: RunContext<any>, value: unknown): Promise<GuardrailResult> {
  const r = await g.check(ctx, value);
  if (typeof r === "boolean") return { tripwire: r };
  return r ?? { tripwire: false };
}

export const inputGuardrail = (name: string, check: GuardFn): Guardrail => ({ name, stage: "input", check });
export const outputGuardrail = (name: string, check: GuardFn): Guardrail => ({ name, stage: "output", check });

/** Block text longer than `chars` characters. */
export function maxLength(chars: number, stage: "input" | "output" = "input"): Guardrail {
  return {
    name: `maxLength(${chars})`,
    stage,
    check: (_ctx, value) => {
      const n = String(value).length;
      return { tripwire: n > chars, info: { length: n, limit: chars } };
    },
  };
}

/** Block text matching any regular expression. */
export function blockedPatterns(patterns: (string | RegExp)[], stage: "input" | "output" = "input"): Guardrail {
  const compiled = patterns.map((p) => (typeof p === "string" ? new RegExp(p, "i") : p));
  return {
    name: "blockedPatterns",
    stage,
    check: (_ctx, value) => {
      const hits = compiled.filter((r) => r.test(String(value))).map((r) => r.source);
      return { tripwire: hits.length > 0, info: { matched: hits } };
    },
  };
}

export const PII_PATTERNS: Record<string, string> = {
  email: "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}",
  phone: "(?<!\\d)(?:\\+?\\d{1,3}[\\s.-]?)?\\(?\\d{3}\\)?[\\s.-]?\\d{3}[\\s.-]?\\d{4}(?!\\d)",
  credit_card: "(?<!\\d)(?:\\d[ -]?){13,19}(?!\\d)",
  ssn: "(?<!\\d)\\d{3}-\\d{2}-\\d{4}(?!\\d)",
  ip_address: "(?<!\\d)(?:\\d{1,3}\\.){3}\\d{1,3}(?!\\d)",
};

/** Detect common PII. `action: "redact"` masks it, `"block"` stops the run. */
export function pii(options: { action?: "redact" | "block"; kinds?: string[]; stage?: "input" | "output" } = {}): Guardrail {
  const action = options.action ?? "redact";
  const selected = Object.entries(PII_PATTERNS).filter(([k]) => !options.kinds || options.kinds.includes(k));
  return {
    name: `pii(${action})`,
    stage: options.stage ?? "input",
    check: (_ctx, value) => {
      if (typeof value !== "string") return { tripwire: false };
      const found: Record<string, number> = {};
      let text = value;
      for (const [kind, pattern] of selected) {
        text = text.replace(new RegExp(pattern, "g"), () => {
          found[kind] = (found[kind] ?? 0) + 1;
          return `[${kind.toUpperCase()}]`;
        });
      }
      if (!Object.keys(found).length) return { tripwire: false };
      if (action === "block") return { tripwire: true, info: { found } };
      return { tripwire: false, info: { found }, replacement: text };
    },
  };
}

/**
 * Use another agent as a classifier. Its `outputType` must include a boolean
 * `tripwire` field (and optionally `reason`).
 */
export function llmGuardrail(agent: Agent<any>, options: { stage?: "input" | "output"; name?: string } = {}): Guardrail {
  return {
    name: options.name ?? "llmGuardrail",
    stage: options.stage ?? "input",
    check: async (ctx, value) => {
      const { run } = await import("./runner.js");
      const result = await run(agent, String(value), { deps: ctx.deps });
      const out = result.output as any;
      return { tripwire: Boolean(out?.tripwire), info: out };
    },
  };
}
