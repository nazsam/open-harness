/**
 * Model providers and the `getModel` resolver. Model strings use `provider:model`:
 *
 *   openai:gpt-6-luna    anthropic:claude-sonnet-5    gemini:gemini-3.8-flash
 *   ollama:llama3.2      groq:<model>                 openrouter:<vendor>/<model>
 *
 * With no provider prefix, the provider is inferred from the model name.
 */

import { ConfigError } from "../errors.js";
import { AnthropicModel } from "./anthropic.js";
import { env, type HTTPOptions, type Model } from "./base.js";
import { FakeModel } from "./fake.js";
import { GeminiModel } from "./gemini.js";
import { COMPATIBLE_PRESETS, OpenAIModel } from "./openai.js";

export const DEFAULT_MODELS: Record<string, string> = {
  openai: "gpt-6-luna",
  anthropic: "claude-sonnet-5",
  gemini: "gemini-3.8-flash",
};

export const PROVIDERS = ["openai", "anthropic", "gemini", "fake", ...Object.keys(COMPATIBLE_PRESETS)];

function inferProvider(name: string): string {
  const n = name.toLowerCase();
  if (n.startsWith("claude")) return "anthropic";
  if (n.startsWith("gemini") || n.startsWith("models/gemini")) return "gemini";
  if (/^(gpt|o1|o3|o4|chatgpt)/.test(n)) return "openai";
  throw new ConfigError(
    `Cannot infer the provider for model '${name}'. Use 'provider:model', for example 'openai:${name}'. Known providers: ${PROVIDERS.join(", ")}`,
  );
}

/** Pick a model from the environment: OPENHARNESS_MODEL, else the first provider with an API key. */
export function defaultModelString(): string {
  const fromEnv = env("OPENHARNESS_MODEL");
  if (fromEnv) return fromEnv;
  for (const [key, provider] of [
    ["ANTHROPIC_API_KEY", "anthropic"],
    ["OPENAI_API_KEY", "openai"],
    ["GEMINI_API_KEY", "gemini"],
    ["GOOGLE_API_KEY", "gemini"],
  ]) {
    if (env(key)) return `${provider}:${DEFAULT_MODELS[provider]}`;
  }
  throw new ConfigError(
    "No model configured. Pass model: 'provider:model', set OPENHARNESS_MODEL, or set one of ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.",
  );
}

/** Resolve a model string (or pass through a `Model` instance). */
export function getModel(spec?: string | Model, options: HTTPOptions & Record<string, unknown> = {}): Model {
  if (spec && typeof spec === "object") return spec;
  const str = spec || defaultModelString();
  const i = str.indexOf(":");
  let provider = i === -1 ? inferProvider(str) : str.slice(0, i).toLowerCase();
  let name = i === -1 ? str : str.slice(i + 1);
  if (!name) name = DEFAULT_MODELS[provider] ?? "";
  if (provider === "google") provider = "gemini";
  switch (provider) {
    case "openai":
      return new OpenAIModel(name, options);
    case "anthropic":
      return new AnthropicModel(name, options);
    case "gemini":
      return new GeminiModel(name, options);
    case "fake":
      return new FakeModel([], { model: name || "fake-model" });
  }
  const preset = COMPATIBLE_PRESETS[provider];
  if (preset) {
    const [baseUrl, keyEnv, required] = preset;
    return new OpenAIModel(name, {
      baseUrl: env(`${provider.toUpperCase()}_BASE_URL`) ?? baseUrl,
      providerName: provider,
      apiKeyEnv: keyEnv,
      requireKey: required,
      ...options,
    });
  }
  throw new ConfigError(`Unknown provider '${provider}'. Known providers: ${PROVIDERS.join(", ")}`);
}

export { AnthropicModel, COMPATIBLE_PRESETS, FakeModel, GeminiModel, OpenAIModel };
export { call } from "./fake.js";
export type { Model, ModelEvent, FetchLike, HTTPOptions } from "./base.js";
