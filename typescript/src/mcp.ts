/**
 * Model Context Protocol (MCP) client.
 *
 *   const fs = new MCPServerStdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]);
 *   const remote = new MCPServerHTTP("https://example.com/mcp", { headers: { Authorization: "Bearer ..." } });
 *   const agent = new Agent({ mcpServers: [fs, remote] });
 *
 * Targets MCP revision 2026-07-28 (stateless, per-request metadata) and falls
 * back to the legacy `initialize` handshake (2025-11-25 and earlier) when a
 * server does not speak the modern protocol, following the spec's dual-era rules.
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { MCPError, ToolError } from "./errors.js";
import { parseSSE, type FetchLike } from "./models/base.js";
import type { Tool } from "./tools.js";
import type { JSONSchema } from "./types.js";
import { VERSION } from "./version.js";

export const MODERN_VERSION = "2026-07-28";
export const LEGACY_VERSION = "2025-11-25";
const CLIENT_INFO = { name: "openharness", version: VERSION };
const MODERN_ERROR_CODES = new Set([-32020, -32021, -32022]);

type RPCMessage = { jsonrpc: "2.0"; id?: number | string; method?: string; params?: any; result?: any; error?: { code: number; message: string; data?: any } };

const isModernError = (msg: any): boolean => typeof msg === "object" && msg !== null && MODERN_ERROR_CODES.has(msg.error?.code);

/** Encode a header value using the spec's Base64 sentinel when needed. */
export function safeHeader(value: string): string {
  const plain = /^[\x20-\x7E\t]*$/.test(value) && value === value.trim();
  const sentinel = value.startsWith("=?base64?") && value.endsWith("?=");
  if (plain && !sentinel) return value;
  return `=?base64?${Buffer.from(value, "utf8").toString("base64")}?=`;
}

const TOKEN = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;

/** Collect `x-mcp-header` annotations. Returns null when the tool definition is invalid. */
export function headerParams(schema: JSONSchema): [string[], string][] | null {
  const found: [string[], string][] = [];
  const seen = new Set<string>();
  const walk = (node: any, path: string[]): boolean => {
    for (const [key, sub] of Object.entries<any>(node?.properties ?? {})) {
      if (!sub || typeof sub !== "object") continue;
      const h = sub["x-mcp-header"];
      if (h !== undefined) {
        if (typeof h !== "string" || !h || !TOKEN.test(h) || seen.has(h.toLowerCase()) || !["string", "integer", "boolean"].includes(sub.type)) return false;
        seen.add(h.toLowerCase());
        found.push([[...path, key], h]);
      }
      if (sub.type === "object" && !walk(sub, [...path, key])) return false;
    }
    return true;
  };
  return walk(schema, []) ? found : null;
}

function contentToText(result: any): string {
  const parts: string[] = [];
  for (const c of result?.content ?? []) {
    if (c.type === "text") parts.push(c.text ?? "");
    else if (c.type === "image" || c.type === "audio") parts.push(`[${c.type} ${c.mimeType ?? ""}]`);
    else if (c.type === "resource") parts.push(c.resource?.text ?? `[resource ${c.resource?.uri ?? ""}]`);
    else if (c.type === "resource_link") parts.push(`[resource ${c.uri ?? ""}]`);
  }
  if (!parts.length && result?.structuredContent !== undefined) return JSON.stringify(result.structuredContent);
  return parts.join("\n");
}

export interface MCPServerOptions {
  name?: string;
  cacheTools?: boolean;
  toolFilter?: (name: string) => boolean;
  toolPrefix?: string;
  timeoutSeconds?: number;
  needsApproval?: boolean;
}

export abstract class MCPServer {
  name: string;
  era?: "modern" | "legacy";
  protocolVersion?: string;
  serverInfo: Record<string, unknown> = {};
  protected tools?: Tool[];
  protected toolDefs = new Map<string, any>();
  protected nextId = 0;
  private connecting?: Promise<void>;

  constructor(
    defaultName: string,
    protected readonly options: MCPServerOptions = {},
  ) {
    this.name = options.name ?? defaultName;
  }

  protected get timeoutMs(): number {
    return (this.options.timeoutSeconds ?? 60) * 1000;
  }

  protected meta(): Record<string, unknown> {
    return {
      "io.modelcontextprotocol/protocolVersion": this.protocolVersion ?? MODERN_VERSION,
      "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
      "io.modelcontextprotocol/clientCapabilities": {},
    };
  }

  protected abstract detect(): Promise<void>;
  protected abstract send(method: string, params: Record<string, unknown>, headers?: Record<string, string>): Promise<any>;
  abstract close(): Promise<void>;

  async connect(): Promise<void> {
    if (this.era) return;
    this.connecting ??= this.detect().finally(() => (this.connecting = undefined));
    await this.connecting;
  }

  async request(method: string, params: Record<string, unknown> = {}, headers?: Record<string, string>): Promise<any> {
    await this.connect();
    const p = { ...params };
    if (this.era === "modern") p._meta = { ...((p._meta as object) ?? {}), ...this.meta() };
    const result = await this.send(method, p, headers);
    if (result && (result.resultType ?? "complete") === "input_required")
      throw new MCPError(`MCP server '${this.name}' asked for extra client input (elicitation/sampling), which this client does not provide.`);
    return result;
  }

  async listTools(): Promise<Tool[]> {
    if (this.tools && this.options.cacheTools !== false) return this.tools;
    const defs: any[] = [];
    let cursor: string | undefined;
    do {
      const res = await this.request("tools/list", cursor ? { cursor } : {});
      defs.push(...(res?.tools ?? []));
      cursor = res?.nextCursor;
    } while (cursor);
    this.toolDefs.clear();
    const tools: Tool[] = [];
    for (const d of defs) {
      if (this.options.toolFilter && !this.options.toolFilter(d.name)) continue;
      if (this instanceof MCPServerHTTP && headerParams(d.inputSchema ?? {}) === null) continue; // spec: exclude invalid x-mcp-header tools
      this.toolDefs.set(d.name, d);
      tools.push(this.wrap(d));
    }
    this.tools = tools;
    return tools;
  }

  invalidateToolsCache(): void {
    this.tools = undefined;
  }

  callTool(name: string, args: Record<string, unknown>): Promise<any> {
    return this.request("tools/call", { name, arguments: args }, this.callHeaders(name, args));
  }

  protected callHeaders(_name: string, _args: Record<string, unknown>): Record<string, string> {
    return {};
  }

  private wrap(d: any): Tool {
    return {
      name: `${this.options.toolPrefix ?? ""}${d.name}`,
      description: d.description ?? d.title ?? "",
      parameters: d.inputSchema ?? { type: "object", properties: {} },
      needsApproval: this.options.needsApproval,
      metadata: { mcpServer: this.name, annotations: d.annotations },
      execute: async (args) => {
        const res = await this.callTool(d.name, args);
        const text = contentToText(res);
        if (res?.isError) throw new ToolError(text || "MCP tool reported an error");
        return text;
      },
    };
  }
}

// ----------------------------------------------------------------------- stdio

export class MCPServerStdio extends MCPServer {
  private proc?: ChildProcessWithoutNullStreams;
  private pending = new Map<number, { resolve: (m: RPCMessage) => void; reject: (e: Error) => void }>();
  stderrTail: string[] = [];

  constructor(
    readonly command: string,
    readonly args: string[] = [],
    readonly spawnOptions: MCPServerOptions & { env?: Record<string, string>; cwd?: string; probeTimeoutSeconds?: number } = {},
  ) {
    super(command.split(/[\\/]/).pop() ?? "mcp", spawnOptions);
  }

  private start(): void {
    if (this.proc && this.proc.exitCode === null) return;
    const proc = spawn(this.command, this.args, {
      env: { ...process.env, ...(this.spawnOptions.env ?? {}) },
      cwd: this.spawnOptions.cwd,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.proc = proc;
    createInterface({ input: proc.stdout }).on("line", (line) => this.onLine(line));
    createInterface({ input: proc.stderr }).on("line", (line) => {
      this.stderrTail = [...this.stderrTail, line].slice(-20);
    });
    const fail = (why: string) => {
      const err = new MCPError(`MCP server '${this.name}' ${why}. stderr: ${this.stderrTail.slice(-5).join(" | ")}`);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
    proc.on("exit", () => fail("exited"));
    proc.on("error", (e) => fail(`failed to start (${e.message})`));
  }

  private onLine(line: string): void {
    if (!line.trim()) return;
    let msg: RPCMessage;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    if (msg.id !== undefined && ("result" in msg || "error" in msg)) {
      const p = this.pending.get(msg.id as number);
      this.pending.delete(msg.id as number);
      p?.resolve(msg);
    } else if (msg.id !== undefined && msg.method) {
      // Legacy servers may send requests (ping, roots/list).
      this.write(msg.method === "ping" ? { jsonrpc: "2.0", id: msg.id, result: {} } : { jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "Method not supported by client" } });
    }
  }

  private write(msg: RPCMessage): void {
    this.proc?.stdin.write(JSON.stringify(msg) + "\n");
  }

  private rpc(method: string, params: Record<string, unknown>, timeoutMs = this.timeoutMs): Promise<RPCMessage> {
    this.start();
    const id = ++this.nextId;
    return new Promise<RPCMessage>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        this.write({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: id, reason: "timeout" } });
        reject(new MCPError(`MCP ${method} timed out`, -1));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (m) => {
          clearTimeout(timer);
          resolve(m);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      this.write({ jsonrpc: "2.0", id, method, params });
    });
  }

  protected async detect(): Promise<void> {
    this.start();
    this.protocolVersion = MODERN_VERSION;
    let msg: RPCMessage | undefined;
    try {
      msg = await this.rpc("server/discover", { _meta: this.meta() }, (this.spawnOptions.probeTimeoutSeconds ?? 5) * 1000);
    } catch (e) {
      if (!(e instanceof MCPError && e.code === -1)) throw e; // only timeouts mean "legacy"
    }
    if (msg?.result) {
      const supported: string[] = msg.result.supportedVersions ?? [MODERN_VERSION];
      this.era = "modern";
      this.protocolVersion = supported.includes(MODERN_VERSION) ? MODERN_VERSION : supported[0];
      this.serverInfo = msg.result.serverInfo ?? {};
      return;
    }
    if (msg && isModernError(msg)) {
      const supported: string[] = msg.error?.data?.supported ?? [];
      const modern = supported.filter((v) => v >= MODERN_VERSION);
      if (!modern.length) throw new MCPError(`MCP server '${this.name}' supports no compatible version: ${supported.join(", ")}`);
      this.era = "modern";
      this.protocolVersion = modern[0];
      return;
    }
    const init = await this.rpc("initialize", { protocolVersion: LEGACY_VERSION, capabilities: {}, clientInfo: CLIENT_INFO });
    if (init.error) throw new MCPError(`MCP initialize failed: ${init.error.message}`, init.error.code, init.error.data);
    this.era = "legacy";
    this.protocolVersion = init.result.protocolVersion ?? LEGACY_VERSION;
    this.serverInfo = init.result.serverInfo ?? {};
    this.write({ jsonrpc: "2.0", method: "notifications/initialized" });
  }

  protected async send(method: string, params: Record<string, unknown>): Promise<any> {
    const msg = await this.rpc(method, params);
    if (msg.error) throw new MCPError(`MCP ${method} failed: ${msg.error.message}`, msg.error.code, msg.error.data);
    return msg.result ?? {};
  }

  async close(): Promise<void> {
    const proc = this.proc;
    this.proc = undefined;
    this.era = undefined;
    this.tools = undefined;
    if (!proc || proc.exitCode !== null) return;
    proc.stdin.end();
    const exited = new Promise<void>((r) => proc.once("exit", () => r()));
    const timer = setTimeout(() => proc.kill("SIGTERM"), 3000);
    await exited;
    clearTimeout(timer);
  }
}

// ------------------------------------------------------------------------ HTTP

export class MCPServerHTTP extends MCPServer {
  private sessionId?: string;
  private readonly fetchImpl: FetchLike;
  private readonly headers: Record<string, string>;

  constructor(
    readonly url: string,
    options: MCPServerOptions & { headers?: Record<string, string>; fetch?: FetchLike } = {},
  ) {
    super(new URL(url).host, options);
    this.headers = options.headers ?? {};
    this.fetchImpl = options.fetch ?? ((i, init) => fetch(i, init));
  }

  protected override callHeaders(name: string, args: Record<string, unknown>): Record<string, string> {
    const out: Record<string, string> = {};
    if (this.era !== "modern") return out;
    for (const [path, header] of headerParams(this.toolDefs.get(name)?.inputSchema ?? {}) ?? []) {
      let value: any = args;
      for (const k of path) value = value && typeof value === "object" ? value[k] : undefined;
      if (value === undefined || value === null) continue;
      out[`Mcp-Param-${header}`] = safeHeader(typeof value === "boolean" ? String(value) : String(value));
    }
    return out;
  }

  private async post(body: RPCMessage, extra: Record<string, string>): Promise<{ status: number; msg: any; headers: Headers }> {
    const resp = await this.fetchImpl(this.url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json, text/event-stream", ...this.headers, ...extra },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    const ctype = resp.headers.get("content-type") ?? "";
    if (ctype.includes("text/event-stream") && resp.body) {
      for await (const { data } of parseSSE(resp.body)) {
        let m: any;
        try {
          m = JSON.parse(data);
        } catch {
          continue;
        }
        if (m.id === body.id && ("result" in m || "error" in m)) {
          await resp.body.cancel().catch(() => undefined);
          return { status: resp.status, msg: m, headers: resp.headers };
        }
      }
      return { status: resp.status, msg: undefined, headers: resp.headers };
    }
    const text = await resp.text();
    let msg: any = text;
    try {
      msg = text ? JSON.parse(text) : undefined;
    } catch {
      /* not JSON */
    }
    return { status: resp.status, msg, headers: resp.headers };
  }

  private modernHeaders(method: string, params: Record<string, unknown>): Record<string, string> {
    const h: Record<string, string> = { "MCP-Protocol-Version": this.protocolVersion ?? MODERN_VERSION, "Mcp-Method": method };
    const name = method === "tools/call" || method === "prompts/get" ? params.name : method === "resources/read" ? params.uri : undefined;
    if (name) h["Mcp-Name"] = safeHeader(String(name));
    return h;
  }

  private legacyHeaders(): Record<string, string> {
    const h: Record<string, string> = { "MCP-Protocol-Version": this.protocolVersion ?? LEGACY_VERSION };
    if (this.sessionId) h["Mcp-Session-Id"] = this.sessionId;
    return h;
  }

  protected async detect(): Promise<void> {
    this.protocolVersion = MODERN_VERSION;
    const params = { _meta: this.meta() };
    let res;
    try {
      res = await this.post({ jsonrpc: "2.0", id: ++this.nextId, method: "server/discover", params }, this.modernHeaders("server/discover", params));
    } catch (e) {
      throw new MCPError(`Cannot reach MCP server at ${this.url}: ${(e as Error).message}`);
    }
    const { status, msg } = res;
    if (status < 400 && msg?.result) {
      const supported: string[] = msg.result.supportedVersions ?? [MODERN_VERSION];
      this.era = "modern";
      this.protocolVersion = supported.includes(MODERN_VERSION) ? MODERN_VERSION : supported[0];
      this.serverInfo = msg.result.serverInfo ?? {};
      return;
    }
    if (isModernError(msg)) {
      const supported: string[] = msg.error?.data?.supported ?? [];
      const modern = supported.filter((v) => v >= MODERN_VERSION);
      if (!modern.length) throw new MCPError(`MCP server '${this.name}' supports no compatible version: ${supported.join(", ")}`);
      this.era = "modern";
      this.protocolVersion = modern[0];
      return;
    }
    if (status === 404 && msg?.error?.code === -32601) {
      this.era = "modern";
      return;
    }
    const init = await this.post(
      { jsonrpc: "2.0", id: ++this.nextId, method: "initialize", params: { protocolVersion: LEGACY_VERSION, capabilities: {}, clientInfo: CLIENT_INFO } },
      {},
    );
    if (init.status >= 400 || !init.msg?.result)
      throw new MCPError(`MCP server at ${this.url} rejected both modern and legacy connections (HTTP ${init.status}): ${JSON.stringify(init.msg)}`);
    this.era = "legacy";
    this.protocolVersion = init.msg.result.protocolVersion ?? LEGACY_VERSION;
    this.serverInfo = init.msg.result.serverInfo ?? {};
    this.sessionId = init.headers.get("mcp-session-id") ?? undefined;
    await this.post({ jsonrpc: "2.0", method: "notifications/initialized" }, this.legacyHeaders());
  }

  protected async send(method: string, params: Record<string, unknown>, headers: Record<string, string> = {}): Promise<any> {
    const extra = this.era === "modern" ? { ...this.modernHeaders(method, params), ...headers } : this.legacyHeaders();
    const { status, msg } = await this.post({ jsonrpc: "2.0", id: ++this.nextId, method, params }, extra);
    if (!msg || typeof msg !== "object") throw new MCPError(`MCP ${method}: unexpected HTTP ${status} response: ${String(msg).slice(0, 300)}`);
    if (msg.error) throw new MCPError(`MCP ${method} failed: ${msg.error.message}`, msg.error.code, msg.error.data);
    return msg.result ?? {};
  }

  async close(): Promise<void> {
    if (this.era === "legacy" && this.sessionId) {
      await this.fetchImpl(this.url, { method: "DELETE", headers: { ...this.headers, ...this.legacyHeaders() } }).catch(() => undefined);
    }
    this.era = undefined;
    this.tools = undefined;
    this.sessionId = undefined;
  }
}
