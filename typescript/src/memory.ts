/**
 * Long-term memory: facts an agent keeps across sessions.
 *
 * Give an agent a memory store and it gets `remember` and `recall` tools, and
 * the most relevant memories are added to its instructions at the start of
 * every run. The built-in stores rank by keyword overlap (BM25). Implement
 * `Memory` to plug in a vector database.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import { s } from "./schema.js";
import { tool, type Tool } from "./tools.js";
import { newId } from "./types.js";

export interface MemoryItem {
  id: string;
  text: string;
  metadata: Record<string, unknown>;
  createdAt: number;
  score?: number;
}

export interface Memory {
  add(text: string, metadata?: Record<string, unknown>): Promise<MemoryItem>;
  search(query: string, limit?: number): Promise<MemoryItem[]>;
  list(): Promise<MemoryItem[]>;
  delete(id: string): Promise<boolean>;
  clear(): Promise<void>;
}

const STOP = new Set(
  "a an and are as at be by for from has have i in is it its of on or that the this to was were will with you your my me we our what who when where how do does did".split(
    " ",
  ),
);

/** Very light stemming so "likes"/"liked"/"liking" match "like". */
function stem(word: string): string {
  for (const suffix of ["ing", "ed", "es", "s"]) {
    if (word.length > suffix.length + 2 && word.endsWith(suffix)) {
      word = word.slice(0, -suffix.length);
      break;
    }
  }
  return word.length > 3 && word.endsWith("e") ? word.slice(0, -1) : word;
}

export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => !STOP.has(w)).map(stem);
}

/** BM25 ranking. Returns the newest items when the query has no keywords. */
export function rank(items: MemoryItem[], query: string, limit: number): MemoryItem[] {
  const q = tokenize(query);
  if (!q.length) return [...items].sort((a, b) => b.createdAt - a.createdAt).slice(0, limit);
  const docs = items.map((i) => tokenize(i.text));
  const n = docs.length || 1;
  const avg = docs.reduce((a, d) => a + d.length, 0) / n || 1;
  const df = new Map<string, number>();
  for (const d of docs) for (const w of new Set(d)) df.set(w, (df.get(w) ?? 0) + 1);
  const scored: MemoryItem[] = [];
  items.forEach((item, idx) => {
    const d = docs[idx];
    let score = 0;
    for (const w of q) {
      const tf = d.filter((x) => x === w).length;
      if (!tf) continue;
      const dfw = df.get(w)!;
      const idf = Math.log(1 + (n - dfw + 0.5) / (dfw + 0.5));
      score += (idf * tf * 2.2) / (tf + 1.2 * (0.25 + (0.75 * d.length) / avg));
    }
    if (score > 0) scored.push({ ...item, score: Math.round(score * 1e4) / 1e4 });
  });
  scored.sort((a, b) => b.score! - a.score! || b.createdAt - a.createdAt);
  return scored.slice(0, limit);
}

export class InMemoryMemory implements Memory {
  protected items: MemoryItem[] = [];

  async add(text: string, metadata: Record<string, unknown> = {}): Promise<MemoryItem> {
    const existing = this.items.find((i) => i.text.trim().toLowerCase() === text.trim().toLowerCase());
    if (existing) return existing;
    const item: MemoryItem = { id: newId("mem"), text: text.trim(), metadata, createdAt: Date.now() / 1000 };
    this.items.push(item);
    return item;
  }

  async search(query: string, limit = 5): Promise<MemoryItem[]> {
    return rank(this.items, query, limit);
  }

  async list(): Promise<MemoryItem[]> {
    return [...this.items];
  }

  async delete(id: string): Promise<boolean> {
    const before = this.items.length;
    this.items = this.items.filter((i) => i.id !== id);
    return this.items.length < before;
  }

  async clear(): Promise<void> {
    this.items = [];
  }
}

/** Memory persisted to a JSON file (same format as the Python SDK). */
export class FileMemory extends InMemoryMemory {
  constructor(readonly path: string) {
    super();
    if (existsSync(path)) {
      const data = JSON.parse(readFileSync(path, "utf8") || "[]") as any[];
      this.items = data.map((d) => ({ id: d.id, text: d.text, metadata: d.metadata ?? {}, createdAt: d.created_at ?? d.createdAt }));
    }
  }

  private async save(): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true });
    const payload = this.items.map((i) => ({ text: i.text, id: i.id, metadata: i.metadata, created_at: i.createdAt }));
    await writeFile(this.path, JSON.stringify(payload, null, 2), "utf8");
  }

  override async add(text: string, metadata: Record<string, unknown> = {}): Promise<MemoryItem> {
    const item = await super.add(text, metadata);
    await this.save();
    return item;
  }

  override async delete(id: string): Promise<boolean> {
    const ok = await super.delete(id);
    await this.save();
    return ok;
  }

  override async clear(): Promise<void> {
    await super.clear();
    await this.save();
  }

  /** Reload from disk (useful when another process wrote to the file). */
  async reload(): Promise<void> {
    const data = JSON.parse((await readFile(this.path, "utf8")) || "[]") as any[];
    this.items = data.map((d) => ({ id: d.id, text: d.text, metadata: d.metadata ?? {}, createdAt: d.created_at ?? d.createdAt }));
  }
}

/** The `remember` and `recall` tools given to agents that have memory. */
export function memoryTools(memory: Memory): Tool[] {
  return [
    tool({
      name: "remember",
      description:
        "Save a durable fact about the user or task for future conversations. Only store stable, useful facts (preferences, names, decisions), never secrets.",
      parameters: s.object({ fact: s.string().describe("One short, self-contained sentence.") }),
      execute: async ({ fact }) => `Saved memory ${(await memory.add(fact)).id}`,
    }),
    tool({
      name: "recall",
      description: "Search long-term memory for facts related to a query.",
      parameters: s.object({ query: s.string().describe("What to look for.") }),
      execute: async ({ query }) => (await memory.search(query, 5)).map((i) => i.text),
    }),
  ];
}
