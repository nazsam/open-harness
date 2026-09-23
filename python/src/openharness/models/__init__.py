"""Model providers and the ``get_model`` resolver.

Model strings use ``provider:model``:

    openai:gpt-6-luna          anthropic:claude-sonnet-5     gemini:gemini-3.8-flash
    ollama:llama3.2            groq:<model>                  openrouter:<vendor>/<model>

With no provider prefix, the provider is inferred from the model name.
"""

from __future__ import annotations

import os

from ..errors import ConfigError
from .anthropic import AnthropicModel
from .base import HTTPClient, Model, ModelEvent
from .fake import FakeModel, call
from .gemini import GeminiModel
from .openai import COMPATIBLE_PRESETS, OpenAIModel

DEFAULT_MODELS = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
}

PROVIDERS = ["openai", "anthropic", "gemini", "fake", *COMPATIBLE_PRESETS]


def _infer_provider(name: str) -> str:
    n = name.lower()
    if n.startswith("claude"):
        return "anthropic"
    if n.startswith("gemini") or n.startswith("models/gemini"):
        return "gemini"
    if n.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    raise ConfigError(
        f"Cannot infer the provider for model '{name}'. Use 'provider:model', "
        f"for example 'openai:{name}'. Known providers: {', '.join(PROVIDERS)}"
    )


def default_model_string() -> str:
    """Pick a model from the environment: OPENHARNESS_MODEL, else the first provider with an API key."""
    if os.environ.get("OPENHARNESS_MODEL"):
        return os.environ["OPENHARNESS_MODEL"]
    for env, provider in (
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("GEMINI_API_KEY", "gemini"),
        ("GOOGLE_API_KEY", "gemini"),
    ):
        if os.environ.get(env):
            return f"{provider}:{DEFAULT_MODELS[provider]}"
    raise ConfigError(
        "No model configured. Pass model='provider:model', set OPENHARNESS_MODEL, or set one of "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY."
    )


def get_model(spec: str | Model | None = None, **kwargs: object) -> Model:
    """Resolve a model string (or pass through a ``Model`` instance)."""
    if isinstance(spec, Model):
        return spec
    spec = spec or default_model_string()
    provider, sep, name = spec.partition(":")
    if not sep:
        provider, name = _infer_provider(spec), spec
    provider = provider.lower()
    if not name:
        name = DEFAULT_MODELS.get(provider, "")
    if provider == "openai":
        return OpenAIModel(name, **kwargs)  # type: ignore[arg-type]
    if provider == "anthropic":
        return AnthropicModel(name, **kwargs)  # type: ignore[arg-type]
    if provider in ("gemini", "google"):
        return GeminiModel(name, **kwargs)  # type: ignore[arg-type]
    if provider == "fake":
        return FakeModel(model=name or "fake-model")
    if provider in COMPATIBLE_PRESETS:
        base_url, key_env, required = COMPATIBLE_PRESETS[provider]
        env_base = os.environ.get(f"{provider.upper()}_BASE_URL")
        opts = {
            "base_url": env_base or base_url,
            "provider_name": provider,
            "api_key_env": key_env,
            "require_key": required,
            **kwargs,
        }
        return OpenAIModel(name, **opts)  # type: ignore[arg-type]
    raise ConfigError(f"Unknown provider '{provider}'. Known providers: {', '.join(PROVIDERS)}")


__all__ = [
    "Model",
    "ModelEvent",
    "HTTPClient",
    "OpenAIModel",
    "AnthropicModel",
    "GeminiModel",
    "FakeModel",
    "call",
    "get_model",
    "default_model_string",
    "DEFAULT_MODELS",
    "PROVIDERS",
]
