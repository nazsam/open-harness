#!/usr/bin/env node
/**
 * Command line interface.
 *
 *   openharness chat --model anthropic:claude-sonnet-5
 *   openharness chat --model ollama:llama3.2 --mcp "npx -y @modelcontextprotocol/server-filesystem ."
 *   openharness run "What is 2**32?" --model openai:gpt-6-luna
 */

import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { ALL_TOOLS } from "./builtinTools.js";
import { OpenHarnessError, RunStopped } from "./errors.js";
import { Harness } from "./harness.js";
import { MCPServerHTTP, MCPServerStdio, type MCPServer } from "./mcp.js";
import { DEFAULT_MODELS, PROVIDERS } from "./models/index.js";
import type { Approver } from "./runner.js";
import { ConsoleProcessor, JSONLProcessor } from "./tracing.js";
import { VERSION } from "./version.js";

const HELP = `openharness ${VERSION}: chat with an AI agent from your terminal

Usage:
  openharness [chat] [options]        interactive chat (default)
  openharness run <prompt> [options]  answer one prompt and exit (stdin is appended)
  openharness models                  list providers and default models

Options:
  -m, --model <provider:model>  e.g. anthropic:claude-sonnet-5, openai:gpt-6-luna, ollama:llama3.2
  -s, --system <text>           system instructions
      --session <file.jsonl>    save the conversation to a file
      --memory <file.json>      long-term memory file (enables remember/recall)
      --tools <list>            comma list from: ${Object.keys(ALL_TOOLS).join(", ")}; or "none"
      --mcp <command>           MCP stdio server command (repeatable)
      --mcp-http <url>          MCP HTTP server URL (repeatable)
      --max-turns <n>           default 20
      --max-tool-calls <n>      default 50
      --timeout <seconds>       per reply, default 600
      --trace <console|file>    print spans or write JSONL traces
  -y, --yes                     auto-approve tools that need approval
  -h, --help / -v, --version`;

/** Split a command string into argv, honouring simple quotes. */
function splitCommand(cmd: string): string[] {
  return (cmd.match(/"[^"]*"|'[^']*'|\S+/g) ?? []).map((p) => p.replace(/^["']|["']$/g, ""));
}

async function readStdin(): Promise<string> {
  if (process.stdin.isTTY) return "";
  const chunks: Buffer[] = [];
  for await (const c of process.stdin) chunks.push(c as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      model: { type: "string", short: "m" },
      system: { type: "string", short: "s" },
      session: { type: "string" },
      memory: { type: "string" },
      tools: { type: "string", default: "calculator,current_time" },
      mcp: { type: "string", multiple: true },
      "mcp-http": { type: "string", multiple: true },
      "max-turns": { type: "string", default: "20" },
      "max-tool-calls": { type: "string", default: "50" },
      timeout: { type: "string", default: "600" },
      trace: { type: "string" },
      yes: { type: "boolean", short: "y", default: false },
      help: { type: "boolean", short: "h", default: false },
      version: { type: "boolean", short: "v", default: false },
    },
  });
  if (values.help) return console.log(HELP), 0;
  if (values.version) return console.log(`openharness ${VERSION}`), 0;
  const [command = "chat", ...rest] = positionals;
  if (command === "models") {
    for (const p of PROVIDERS) console.log(`${p.padEnd(11)} ${DEFAULT_MODELS[p] ?? "(pass a model name)"}`);
    return 0;
  }
  if (command !== "chat" && command !== "run") {
    console.error(`Unknown command '${command}'.\n\n${HELP}`);
    return 1;
  }

  const toolNames = values.tools === "none" ? [] : values.tools!.split(",").map((t) => t.trim()).filter(Boolean);
  const unknown = toolNames.filter((t) => !(t in ALL_TOOLS));
  if (unknown.length) {
    console.error(`Unknown tools: ${unknown.join(", ")}. Available: ${Object.keys(ALL_TOOLS).join(", ")}`);
    return 1;
  }
  const servers: MCPServer[] = [
    ...(values.mcp ?? []).map((c) => {
      const [cmd, ...args] = splitCommand(c);
      return new MCPServerStdio(cmd, args);
    }),
    ...(values["mcp-http"] ?? []).map((u) => new MCPServerHTTP(u)),
  ];
  const approve: Approver | undefined = values.yes ? () => true : undefined;
  const harness = new Harness({
    model: values.model,
    instructions: values.system,
    tools: toolNames.map((t) => ALL_TOOLS[t]),
    session: values.session,
    memory: values.memory,
    mcpServers: servers,
    limits: { maxTurns: Number(values["max-turns"]), maxToolCalls: Number(values["max-tool-calls"]), timeoutSeconds: Number(values.timeout) },
    trace: values.trace ? (values.trace === "console" ? new ConsoleProcessor() : new JSONLProcessor(values.trace)) : undefined,
    approve,
  });

  try {
    if (command === "run") {
      let prompt = rest.join(" ");
      const piped = await readStdin();
      if (piped.trim()) prompt = `${prompt}\n\n${piped}`;
      if (!prompt.trim()) {
        console.error("Nothing to ask. Usage: openharness run <prompt>");
        return 1;
      }
      for await (const chunk of harness.stream(prompt)) process.stdout.write(chunk);
      process.stdout.write("\n");
    } else {
      await harness.chat();
    }
    return 0;
  } catch (e) {
    if (e instanceof RunStopped) return console.error(`stopped: ${e.message}`), 2;
    if (e instanceof OpenHarnessError) return console.error(`error: ${e.message}`), 1;
    throw e;
  } finally {
    await harness.close();
  }
}

function isMain(): boolean {
  try {
    return Boolean(process.argv[1]) && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}

if (isMain()) {
  main().then(
    (code) => process.exit(code),
    (e) => {
      console.error(e);
      process.exit(1);
    },
  );
}
