/**
 * Sessions keep conversation history between runs.
 *
 *   const session = new FileSession("chats/alice.jsonl");
 *   await run(agent, "My name is Alice", { session });
 *   await run(agent, "What is my name?", { session }); // "Alice"
 *
 * Implement `Session` to store history anywhere (Redis, Postgres, your API).
 */

import { appendFile, mkdir, readFile, rm } from "node:fs/promises";
import { dirname } from "node:path";
import { messageFromJSON, messageToJSON, type Message } from "./types.js";

export interface Session {
  getMessages(limit?: number): Promise<Message[]>;
  addMessages(messages: Message[]): Promise<void>;
  clear(): Promise<void>;
}

/** Keep the last `limit` messages without splitting a tool call from its results. */
function trim(messages: Message[], limit?: number): Message[] {
  if (!limit || messages.length <= limit) return [...messages];
  let out = messages.slice(-limit);
  while (out.length && out[0].role === "tool") out = out.slice(1);
  return out;
}

export class InMemorySession implements Session {
  private messages: Message[] = [];
  constructor(readonly maxMessages?: number) {}

  async getMessages(limit?: number): Promise<Message[]> {
    return trim(this.messages, limit ?? this.maxMessages);
  }

  async addMessages(messages: Message[]): Promise<void> {
    this.messages.push(...messages);
  }

  async clear(): Promise<void> {
    this.messages = [];
  }
}

/** History appended to a JSON Lines file. Human readable and easy to back up. */
export class FileSession implements Session {
  private queue: Promise<unknown> = Promise.resolve();
  constructor(
    readonly path: string,
    readonly maxMessages?: number,
  ) {}

  async getMessages(limit?: number): Promise<Message[]> {
    await this.queue;
    let text: string;
    try {
      text = await readFile(this.path, "utf8");
    } catch (e: any) {
      if (e.code === "ENOENT") return [];
      throw e;
    }
    const messages = text
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => messageFromJSON(JSON.parse(l)));
    return trim(messages, limit ?? this.maxMessages);
  }

  addMessages(messages: Message[]): Promise<void> {
    const job = this.queue.then(async () => {
      await mkdir(dirname(this.path), { recursive: true });
      await appendFile(this.path, messages.map((m) => JSON.stringify(messageToJSON(m)) + "\n").join(""), "utf8");
    });
    this.queue = job.catch(() => undefined);
    return job;
  }

  clear(): Promise<void> {
    const job = this.queue.then(() => rm(this.path, { force: true }));
    this.queue = job.catch(() => undefined);
    return job;
  }
}
