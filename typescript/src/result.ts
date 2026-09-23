/** Run results and the streaming handle. */

import type { Agent } from "./agent.js";
import type { CancelToken } from "./context.js";
import type { Trace } from "./tracing.js";
import type { Message, StreamEvent, Usage } from "./types.js";

export interface RunResult<TOutput = string> {
  /** Parsed structured output when the agent has an `outputType`, otherwise the final text. */
  output: TOutput;
  text: string;
  /** Every message created during the run (the input included). */
  newMessages: Message[];
  /** Messages loaded from the session before the run. */
  history: Message[];
  lastAgent: Agent<any>;
  usage: Usage;
  turns: number;
  toolCalls: number;
  trace?: Trace;
}

/** Full conversation (history + this run), ready to pass as the next input. */
export const toMessages = (r: RunResult<any>): Message[] => [...r.history, ...r.newMessages];

/**
 * Handle returned by `runStream`. Iterate for events, then await `result()`.
 *
 *   const stream = runStream(agent, "Tell me a story");
 *   for await (const ev of stream) if (ev.type === "text_delta") process.stdout.write(ev.data.delta);
 *   const result = await stream.result();
 */
export class RunStream<TOutput = string> implements AsyncIterable<StreamEvent> {
  private final?: RunResult<TOutput>;
  private error?: unknown;
  private finished = false;
  private iterator: AsyncIterator<StreamEvent>;

  constructor(
    gen: AsyncGenerator<StreamEvent>,
    private readonly token: CancelToken,
  ) {
    this.iterator = gen;
  }

  async *[Symbol.asyncIterator](): AsyncIterator<StreamEvent> {
    try {
      for (;;) {
        const { value, done } = await this.iterator.next();
        if (done) break;
        if (value.type === "run_end") this.final = value.data.result;
        yield value;
      }
    } catch (e) {
      this.error = e;
      throw e;
    } finally {
      this.finished = true;
    }
  }

  /** Only the text deltas. */
  async *text(): AsyncIterable<string> {
    for await (const ev of this) if (ev.type === "text_delta") yield ev.data.delta as string;
  }

  /** Wait for the run to finish (consuming any remaining events). */
  async result(): Promise<RunResult<TOutput>> {
    if (!this.finished) {
      for await (const _ of this) {
        /* drain */
      }
    }
    if (this.error) throw this.error;
    return this.final!;
  }

  cancel(reason = "cancelled"): void {
    this.token.cancel(reason);
  }
}
