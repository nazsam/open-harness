/**
 * The agent loop.
 *
 *   const result = await run(agent, "What's the weather in Paris?");
 *
 * Each turn: build the prompt, call the model, run any requested tools (in
 * parallel), feed results back, repeat until the model answers without tool
 * calls. Handoffs switch the active agent. Limits, cancellation and
 * guardrails are checked at every step.
 */

import type { Agent, Handoff } from "./agent.js";
import { CancelToken, DEFAULT_LIMITS, RunContext, type RunLimits } from "./context.js";
import {
  GuardrailTripped,
  MaxToolCallsExceeded,
  MaxTurnsExceeded,
  OpenHarnessError,
  OutputValidationError,
  RunCancelled,
  RunStopped,
  RunTimeout,
  TokenBudgetExceeded,
  ToolError,
} from "./errors.js";
import { runGuardrail, type Guardrail } from "./guardrails.js";
import { memoryTools } from "./memory.js";
import type { Model } from "./models/index.js";
import { RunStream, type RunResult } from "./result.js";
import { extractJson, outputSchemaFor, SchemaValidationError, validate } from "./schema.js";
import type { Session } from "./sessions.js";
import { invokeTool, requiresApproval, toolSpec, type Tool } from "./tools.js";
import { globalProcessors, Trace, type Span, type TraceProcessor } from "./tracing.js";
import {
  mergeSettings,
  Message,
  toJSONText,
  type ModelRequest,
  type ModelResponse,
  type ModelSettings,
  type StreamEvent,
  type ToolCall,
} from "./types.js";

export type Approver = (ctx: RunContext<any>, call: ToolCall) => boolean | Promise<boolean>;

export interface RunOptions<TDeps = unknown> {
  session?: Session;
  /** Your own objects (db handles, current user). Passed to tools; never sent to the model. */
  deps?: TDeps;
  /** Override the agent's model for this run. */
  model?: string | Model;
  limits?: RunLimits;
  modelSettings?: ModelSettings;
  /** Standard AbortSignal; aborting it cancels the run. */
  signal?: AbortSignal;
  cancelToken?: CancelToken;
  /** Decides whether tools marked `needsApproval` may run. Without it they are denied. */
  approve?: Approver;
  trace?: TraceProcessor | TraceProcessor[];
  traceMetadata?: Record<string, unknown>;
  /** Called for every stream event (enables streaming from the provider). */
  onEvent?: (event: StreamEvent) => void | Promise<void>;
  /** @internal nested runs (agent as tool) share the parent trace */
  parentTrace?: Trace;
}

type Input = string | Message | Message[];

function inputMessages(input: Input): Message[] {
  if (typeof input === "string") return [Message.user(input)];
  return Array.isArray(input) ? [...input] : [input];
}

async function runGuardrails(list: Guardrail[], ctx: RunContext<any>, value: unknown, stage: "input" | "output", trace: Trace, parent: Span) {
  for (const g of list) {
    const res = await trace.span("guardrail", g.name, parent, { stage }, async (sp) => {
      const r = await runGuardrail(g, ctx, value);
      sp.set({ tripped: r.tripwire });
      return r;
    });
    if (res.tripwire) throw new GuardrailTripped(`${stage} guardrail '${g.name}' tripped`, g.name, stage, res.info);
    if (res.replacement !== undefined) value = res.replacement;
  }
  return value;
}

async function* loop<TOutput>(agent: Agent<TOutput>, input: Input, opts: RunOptions<any>, streaming: boolean): AsyncGenerator<StreamEvent> {
  const limits = { ...DEFAULT_LIMITS, ...(agent.limits ?? {}), ...(opts.limits ?? {}) };
  const token = opts.cancelToken ?? new CancelToken();
  const signals = [token.signal];
  if (opts.signal) signals.push(opts.signal);
  const timeout = limits.timeoutSeconds ? AbortSignal.timeout(limits.timeoutSeconds * 1000) : undefined;
  if (timeout) signals.push(timeout);
  const runSignal = AbortSignal.any(signals);

  const stopError = (): RunStopped => {
    if (token.cancelled) return new RunCancelled(`Run cancelled: ${token.reason}`);
    if (opts.signal?.aborted) return new RunCancelled(`Run cancelled: ${String(opts.signal.reason ?? "aborted")}`);
    return new RunTimeout(`Run exceeded timeout of ${limits.timeoutSeconds}s`);
  };
  const guard = <T>(p: Promise<T>): Promise<T> => {
    if (runSignal.aborted) {
      p.catch(() => undefined);
      return Promise.reject(stopError());
    }
    return new Promise<T>((resolve, reject) => {
      const onAbort = () => reject(stopError());
      runSignal.addEventListener("abort", onAbort, { once: true });
      p.then(
        (v) => {
          runSignal.removeEventListener("abort", onAbort);
          resolve(v);
        },
        (e) => {
          runSignal.removeEventListener("abort", onAbort);
          reject(runSignal.aborted ? stopError() : e);
        },
      );
    });
  };

  const procs = opts.trace ? (Array.isArray(opts.trace) ? opts.trace : [opts.trace]) : [];
  const ownsTrace = !opts.parentTrace;
  const trace = opts.parentTrace ?? new Trace(agent.name, [...procs, ...globalProcessors()], opts.traceMetadata);
  const ctx = new RunContext(agent, opts.deps, trace, runSignal);
  let history = opts.session ? await opts.session.getMessages() : [];
  const inputs = inputMessages(input);
  let fresh: Message[] = [];
  let current: Agent<any> = agent;
  let outputRetries = 0;
  const mcpCache = new Map<object, Tool[]>();

  const partial = (): RunResult<any> => ({
    output: undefined,
    text: "",
    newMessages: fresh,
    history,
    lastAgent: current,
    usage: ctx.usage,
    turns: ctx.turn,
    toolCalls: ctx.toolCalls,
    trace,
  });
  const ev = (type: StreamEvent["type"], data: Record<string, unknown>): StreamEvent => ({ type, data, agent: current.name, timestamp: Date.now() / 1000 });

  const runSpan = trace.startSpan(ownsTrace ? "run" : "agent_tool", agent.name);
  let failure: unknown;
  try {
    if (agent.inputGuardrails.length) {
      for (let i = inputs.length - 1; i >= 0; i--) {
        if (inputs[i].role !== "user") continue;
        const replaced = await guard(runGuardrails(agent.inputGuardrails, ctx, inputs[i].content, "input", trace, runSpan));
        if (replaced !== inputs[i].content) inputs[i] = Message.user(String(replaced));
        break;
      }
    }
    fresh.push(...inputs);
    yield ev("agent_start", { agent: current.name });

    for (;;) {
      if (runSignal.aborted) throw stopError();
      if (limits.maxTurns != null && ctx.turn >= limits.maxTurns) throw new MaxTurnsExceeded(`Agent stopped after reaching maxTurns=${limits.maxTurns}`);
      if (limits.maxTotalTokens != null && ctx.usage.totalTokens >= limits.maxTotalTokens)
        throw new TokenBudgetExceeded(`Token budget of ${limits.maxTotalTokens} used up (${ctx.usage.totalTokens} tokens)`);
      ctx.turn += 1;
      ctx.agent = current;

      // ---- tools
      const tools: Tool[] = [...current.tools];
      if (current.memory) tools.push(...memoryTools(current.memory));
      for (const server of current.mcpServers) {
        if (!mcpCache.has(server)) {
          const list = await trace.span("mcp", server.name, runSpan, { action: "listTools" }, () => guard(server.listTools()));
          mcpCache.set(server, list);
        }
        tools.push(...mcpCache.get(server)!);
      }
      const handoffs = current.handoffObjects();
      tools.push(...handoffs.map((h) => h.tool()));
      const byName = new Map<string, Tool>();
      for (const t of tools) if (!byName.has(t.name)) byName.set(t.name, t);

      // ---- system prompt
      let system = await guard(current.renderInstructions(ctx));
      if (current.memory) {
        const query = [...history, ...fresh].reverse().find((m) => m.role === "user")?.content ?? "";
        const found = await current.memory.search(query, 5);
        if (found.length) system += `\n\nRelevant long-term memories:\n${found.map((m) => `- ${m.text}`).join("\n")}`;
      }

      const model = current.getModel(opts.model);
      let outputSchema: ModelRequest["outputSchema"];
      if (current.outputType) {
        const { schema } = outputSchemaFor(current.outputType);
        const native = !(model.provider === "gemini" && byName.size > 0); // Gemini: schema + tools not combined
        if (native) outputSchema = { name: String(current.outputType.title ?? "output").replace(/[^a-zA-Z0-9_-]/g, "_"), schema };
        else system += `\n\nWhen you give your final answer, reply with only a JSON object that matches this JSON Schema:\n${JSON.stringify(schema)}`;
      }
      const request: ModelRequest = {
        system,
        messages: [...history, ...fresh],
        tools: [...byName.values()].map(toolSpec),
        outputSchema,
        settings: mergeSettings(current.modelSettings, opts.modelSettings),
        signal: runSignal,
      };

      // ---- model call
      const modelSpan = trace.startSpan("model", `${model.provider}:${model.model}`, runSpan, { agent: current.name, turn: ctx.turn });
      let response: ModelResponse | undefined;
      try {
        if (streaming) {
          const it = model.stream(request)[Symbol.asyncIterator]();
          for (;;) {
            const step = await guard(it.next());
            if (step.done) break;
            const e = step.value;
            if (e.type === "text_delta") yield ev("text_delta", { delta: e.delta });
            else if (e.type === "done") response = e.response;
          }
          if (!response) throw new OpenHarnessError(`${model.provider}:${model.model} stream ended without a response`);
        } else {
          response = await guard(model.generate(request));
        }
        modelSpan.set({ usage: response.usage.toJSON(), stopReason: response.stopReason, toolCalls: (response.message.toolCalls ?? []).map((t) => t.name) });
        trace.endSpan(modelSpan);
      } catch (e) {
        trace.endSpan(modelSpan, e);
        throw e;
      }
      ctx.usage.add(response.usage);
      const msg = response.message;
      msg.name = current.name;
      fresh.push(msg);
      ctx.messages = [...history, ...fresh];
      yield ev("message", { message: msg });

      // ---- tool calls
      const calls = msg.toolCalls ?? [];
      if (calls.length) {
        if (limits.maxToolCalls != null && ctx.toolCalls + calls.length > limits.maxToolCalls) {
          fresh.pop();
          throw new MaxToolCallsExceeded(`Agent stopped after reaching maxToolCalls=${limits.maxToolCalls}`);
        }
        ctx.toolCalls += calls.length;
        for (const c of calls) yield ev("tool_call", { id: c.id, name: c.name, arguments: c.arguments });
        let handoffTarget: Handoff | undefined;

        const execute = async (tc: ToolCall): Promise<Message> => {
          const t = byName.get(tc.name);
          if (!t) return Message.tool(tc.id, tc.name, `Error: unknown tool '${tc.name}'. Available: ${[...byName.keys()].join(", ") || "none"}`, true);
          const h = t.metadata?.handoff as Handoff | undefined;
          if (h) {
            if (handoffTarget) return Message.tool(tc.id, tc.name, "Ignored: another handoff was already made.", true);
            handoffTarget = h;
            return Message.tool(tc.id, tc.name, `Transferred to ${h.agent.name}.`);
          }
          const span = trace.startSpan("tool", tc.name, modelSpan, { arguments: tc.arguments });
          try {
            if (requiresApproval(t, tc.arguments)) {
              const ok = opts.approve ? await opts.approve(ctx, tc) : false;
              span.set({ approved: ok });
              if (!ok) return Message.tool(tc.id, tc.name, "Error: the user did not approve this tool call.", true);
            }
            const out = toJSONText(await invokeTool(t, tc.arguments, ctx));
            span.set({ output: out.slice(0, 2000) });
            return Message.tool(tc.id, tc.name, out);
          } catch (e) {
            if (e instanceof RunStopped) throw e;
            span.error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
            const text = e instanceof ToolError ? e.message : e instanceof Error ? `${e.name}: ${e.message}` : String(e);
            return Message.tool(tc.id, tc.name, `Error: ${text}`, true);
          } finally {
            trace.endSpan(span);
          }
        };

        const results = await guard(Promise.all(calls.map(execute)));
        for (const r of results) {
          fresh.push(r);
          yield ev("tool_result", { id: r.toolCallId, name: r.name, output: r.content, isError: Boolean(r.isError) });
        }
        if (handoffTarget) {
          const target: Handoff = handoffTarget;
          const prev = current;
          const hs = trace.startSpan("handoff", `${prev.name} -> ${target.agent.name}`, runSpan);
          current = target.agent;
          if (target.options.inputFilter) {
            fresh = target.options.inputFilter([...history, ...fresh]);
            history = [];
          }
          trace.endSpan(hs);
          yield ev("handoff", { from: prev.name, to: current.name });
          yield ev("agent_start", { agent: current.name });
        }
        continue;
      }

      // ---- final answer
      let text = msg.content;
      let output: unknown = text;
      if (current.outputType) {
        try {
          const { schema, wrapped } = outputSchemaFor(current.outputType);
          const data = extractJson(text);
          validate(data, schema);
          output = wrapped ? (data as any).value : data;
        } catch (e) {
          if (!(e instanceof SchemaValidationError)) throw e;
          if (outputRetries >= limits.maxOutputRetries) throw new OutputValidationError(`Structured output failed validation: ${e.message}`);
          outputRetries += 1;
          fresh.push(Message.user(`Your reply did not match the required JSON schema: ${e.message}. Reply again with only the corrected JSON.`));
          continue;
        }
      }
      if (current.outputGuardrails.length) {
        output = await guard(runGuardrails(current.outputGuardrails, ctx, output, "output", trace, runSpan));
        if (typeof output === "string") text = output;
      }
      const result: RunResult<any> = {
        output,
        text,
        newMessages: fresh,
        history,
        lastAgent: current,
        usage: ctx.usage,
        turns: ctx.turn,
        toolCalls: ctx.toolCalls,
        trace,
      };
      if (opts.session) await opts.session.addMessages(fresh);
      runSpan.set({ turns: ctx.turn, usage: ctx.usage.toJSON(), lastAgent: current.name });
      yield ev("run_end", { result });
      return;
    }
  } catch (e) {
    failure = e;
    if (e instanceof RunStopped && !e.partial) e.partial = partial();
    throw e;
  } finally {
    trace.endSpan(runSpan, failure);
    if (ownsTrace) trace.finish();
  }
}

/** Run an agent to completion and return the result. */
export async function run<TOutput = string, TDeps = unknown>(agent: Agent<TOutput>, input: Input, options: RunOptions<TDeps> = {}): Promise<RunResult<TOutput>> {
  let result: RunResult<TOutput> | undefined;
  for await (const e of loop(agent, input, options, Boolean(options.onEvent))) {
    if (options.onEvent) await options.onEvent(e);
    if (e.type === "run_end") result = e.data.result;
  }
  return result!;
}

/** Start a run and stream events. See `RunStream`. */
export function runStream<TOutput = string, TDeps = unknown>(agent: Agent<TOutput>, input: Input, options: RunOptions<TDeps> = {}): RunStream<TOutput> {
  const token = options.cancelToken ?? new CancelToken();
  return new RunStream<TOutput>(loop(agent, input, { ...options, cancelToken: token }, true), token);
}
