/** Interactive terminal chat used by `Harness.chat()` and the CLI. */

import { createInterface } from "node:readline";
import { OpenHarnessError, RunCancelled, RunStopped } from "./errors.js";
import type { Harness } from "./harness.js";
import { runStream } from "./runner.js";
import { VERSION } from "./version.js";

export async function terminalChat(
  harness: Harness,
  options: { input?: NodeJS.ReadableStream; output?: NodeJS.WritableStream; showTools?: boolean } = {},
): Promise<void> {
  const input = options.input ?? process.stdin;
  const out = options.output ?? process.stdout;
  const showTools = options.showTools ?? true;
  const color = (out as any).isTTY && !process.env.NO_COLOR;
  const [dim, bold, cyan, red, reset] = color ? ["\x1b[2m", "\x1b[1m", "\x1b[36m", "\x1b[31m", "\x1b[0m"] : ["", "", "", "", ""];
  const write = (s: string) => out.write(s);
  const agent = harness.agent;

  let label: string;
  try {
    const m = agent.getModel();
    label = `${m.provider}:${m.model}`;
  } catch (e) {
    write(`${red}${(e as Error).message}${reset}\n`);
    return;
  }
  write(`${bold}openharness ${VERSION}${reset}  model ${cyan}${label}${reset}  tools ${agent.tools.length}  ${dim}(/help for commands, Ctrl+C stops a reply)${reset}\n`);

  const rl = createInterface({ input, output: out, terminal: Boolean((input as any).isTTY) });
  const lines = rl[Symbol.asyncIterator]();
  let active: ReturnType<typeof runStream> | undefined;
  rl.on("SIGINT", () => {
    if (active) active.cancel("interrupted");
    else rl.close();
  });

  const ask = async (prompt: string): Promise<string | undefined> => {
    write(prompt);
    const next = await lines.next();
    return next.done ? undefined : next.value;
  };

  harness.approve ??= async (_ctx, call) => {
    const answer = await ask(`\nAllow tool ${call.name}(${JSON.stringify(call.arguments).slice(0, 300)})? [y/N] `);
    return ["y", "yes"].includes((answer ?? "").trim().toLowerCase());
  };

  for (;;) {
    const line = await ask(`\n${bold}you>${reset} `);
    if (line === undefined) break;
    const text = line.trim();
    if (!text) continue;
    if (text.startsWith("/")) {
      const cmd = text.split(/\s+/)[0].toLowerCase();
      if (["/exit", "/quit", "/q"].includes(cmd)) break;
      if (cmd === "/reset") {
        await harness.reset();
        write(`${dim}conversation cleared${reset}\n`);
      } else if (cmd === "/usage") write(`${dim}${JSON.stringify(harness.usage)}${reset}\n`);
      else if (cmd === "/tools") {
        const names = agent.tools.map((t) => t.name);
        for (const server of agent.mcpServers) names.push(...(await server.listTools()).map((t) => `${t.name} (mcp:${server.name})`));
        write(`${dim}${names.join(", ") || "no tools"}${reset}\n`);
      } else write(`${dim}/reset  clear the conversation\n/usage  token usage\n/tools  list tools\n/exit   quit${reset}\n`);
      continue;
    }
    active = runStream(agent, text, harness.runOptions());
    write(`${bold}${agent.name}>${reset} `);
    try {
      for await (const ev of active) {
        if (ev.type === "text_delta") write(ev.data.delta);
        else if (ev.type === "tool_call" && showTools) write(`\n${dim}  -> ${ev.data.name}(${JSON.stringify(ev.data.arguments).slice(0, 200)})${reset}\n`);
        else if (ev.type === "tool_result" && showTools)
          write(`${dim}  <- ${ev.data.isError ? "error" : "ok"}: ${String(ev.data.output).replace(/\n/g, " ").slice(0, 160)}${reset}\n`);
        else if (ev.type === "handoff") write(`\n${dim}  handoff: ${ev.data.from} -> ${ev.data.to}${reset}\n`);
      }
      const result = await active.result();
      harness.usage.add(result.usage);
      if (typeof result.output !== "string") write(JSON.stringify(result.output, null, 2));
      write("\n");
    } catch (e) {
      if (e instanceof RunCancelled) write(`\n${dim}(stopped)${reset}\n`);
      else if (e instanceof RunStopped) write(`\n${red}stopped: ${e.message}${reset}\n`);
      else if (e instanceof OpenHarnessError) write(`\n${red}error: ${e.message}${reset}\n`);
      else throw e;
    } finally {
      active = undefined;
    }
  }
  rl.close();
}
