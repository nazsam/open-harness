/** A few safe, dependency-free tools that the harness and CLI enable by default. */

import { ToolError } from "./errors.js";
import { s } from "./schema.js";
import { tool, type Tool } from "./tools.js";

const FUNCS: Record<string, (...a: number[]) => number> = {
  sqrt: Math.sqrt, sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  log: Math.log, log10: Math.log10, log2: Math.log2, exp: Math.exp, floor: Math.floor, ceil: Math.ceil, abs: Math.abs,
  round: Math.round, min: Math.min, max: Math.max,
  radians: (d) => (d * Math.PI) / 180, degrees: (r) => (r * 180) / Math.PI,
  factorial: (n) => {
    if (!Number.isInteger(n) || n < 0 || n > 170) throw new ToolError("factorial needs an integer between 0 and 170");
    let r = 1;
    for (let i = 2; i <= n; i++) r *= i;
    return r;
  },
};
const CONSTS: Record<string, number> = { pi: Math.PI, e: Math.E, tau: 2 * Math.PI };

/** Evaluate arithmetic without `eval`: numbers, + - * / // % ** ^, parentheses and math functions. */
export function safeEval(expression: string): number {
  const tokens = expression.match(/\d+\.?\d*(?:e[+-]?\d+)?|\.\d+|\*\*|\/\/|[a-z_][a-z0-9_]*|[-+*/%^(),]|\S/gi) ?? [];
  let pos = 0;
  const peek = () => tokens[pos];
  const take = (t?: string) => {
    const tok = tokens[pos++];
    if (t !== undefined && tok !== t) throw new ToolError(`Could not evaluate '${expression}': expected '${t}'`);
    return tok;
  };
  const expr = (): number => {
    let v = term();
    while (peek() === "+" || peek() === "-") v = take() === "+" ? v + term() : v - term();
    return v;
  };
  const term = (): number => {
    let v = unary();
    for (;;) {
      const op = peek();
      if (op === "*") (take(), (v *= unary()));
      else if (op === "/") (take(), (v /= unary()));
      else if (op === "//") (take(), (v = Math.floor(v / unary())));
      else if (op === "%") (take(), (v %= unary()));
      else return v;
    }
  };
  const unary = (): number => {
    if (peek() === "-") return take(), -unary();
    if (peek() === "+") return take(), unary();
    return power();
  };
  const power = (): number => {
    const base = atom();
    if (peek() === "**" || peek() === "^") {
      take();
      const exp = unary();
      if (Math.abs(exp) > 1000) throw new ToolError("Exponent too large");
      return base ** exp;
    }
    return base;
  };
  const atom = (): number => {
    const t = take();
    if (t === undefined) throw new ToolError(`Could not evaluate '${expression}': unexpected end`);
    if (t === "(") {
      const v = expr();
      take(")");
      return v;
    }
    if (/^[\d.]/.test(t)) return Number(t);
    const name = t.toLowerCase();
    if (name in CONSTS) return CONSTS[name];
    if (name in FUNCS && peek() === "(") {
      take("(");
      const args: number[] = [];
      if (peek() !== ")") {
        args.push(expr());
        while (peek() === ",") (take(), args.push(expr()));
      }
      take(")");
      return FUNCS[name](...args);
    }
    throw new ToolError(`Unsupported expression near '${t}'`);
  };
  const value = expr();
  if (pos < tokens.length) throw new ToolError(`Could not evaluate '${expression}': unexpected '${tokens[pos]}'`);
  if (!Number.isFinite(value)) throw new ToolError(`Could not evaluate '${expression}': result is not finite`);
  return value;
}

export const calculator: Tool = tool({
  name: "calculator",
  description: "Evaluate a math expression exactly. Use this instead of doing arithmetic in your head.",
  parameters: s.object({
    expression: s
      .string()
      .describe('For example "(12.5 * 4) / 3" or "sqrt(2) ** 3". Supports + - * / // % ** and sqrt, log, sin, cos, tan, exp, floor, ceil, abs, round, min, max, pi, e.'),
  }),
  execute: ({ expression }) => {
    const v = safeEval(expression);
    return String(Number.isInteger(v) ? v : Number(v.toPrecision(15)));
  },
});

export const currentTime: Tool = tool({
  name: "current_time",
  description: "Get the current date and time.",
  parameters: s.object({ timezone_name: s.string().describe('IANA time zone such as "America/Toronto" or "Europe/London".').default("UTC") }),
  execute: ({ timezone_name }) => {
    const tz = timezone_name ?? "UTC";
    try {
      return new Intl.DateTimeFormat("en-CA", {
        timeZone: tz,
        weekday: "long",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZoneName: "short",
      }).format(new Date());
    } catch {
      throw new ToolError(`Unknown time zone '${tz}'`);
    }
  },
});

export const fetchUrl: Tool = tool({
  name: "fetch_url",
  description: "Download a web page and return its text (HTML tags removed).",
  parameters: s.object({
    url: s.string().describe("An http or https URL."),
    max_chars: s.integer().describe("Maximum characters to return.").default(8000),
  }),
  needsApproval: true,
  execute: async ({ url, max_chars }) => {
    if (!/^https?:\/\//.test(url)) throw new ToolError("Only http and https URLs are allowed");
    const resp = await fetch(url, { headers: { "User-Agent": "openharness" }, signal: AbortSignal.timeout(20000) });
    let text = await resp.text();
    if ((resp.headers.get("content-type") ?? "").includes("html"))
      text = text.replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>|<[^>]+>/gi, " ").replace(/\s+/g, " ");
    return `HTTP ${resp.status}\n${text.trim().slice(0, max_chars ?? 8000)}`;
  },
});

export const DEFAULT_TOOLS: Tool[] = [calculator, currentTime];
export const ALL_TOOLS: Record<string, Tool> = { calculator, current_time: currentTime, fetch_url: fetchUrl };
