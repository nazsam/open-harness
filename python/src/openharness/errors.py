"""Exception hierarchy. Every SDK error derives from ``OpenHarnessError``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .result import RunResult


class OpenHarnessError(Exception):
    """Base class for all SDK errors."""


class ModelError(OpenHarnessError):
    """The model provider returned an error or an unusable response."""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None, provider: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body
        self.provider = provider


class ConfigError(OpenHarnessError):
    """Invalid configuration, such as an unknown provider or missing API key."""


class ToolError(OpenHarnessError):
    """Raise inside a tool to send a clean error message back to the model."""


class RunStopped(OpenHarnessError):
    """Base class for errors that end a run early. ``partial`` holds what was done so far."""

    def __init__(self, message: str, partial: RunResult | None = None):
        super().__init__(message)
        self.partial = partial


class MaxTurnsExceeded(RunStopped):
    pass


class MaxToolCallsExceeded(RunStopped):
    pass


class TokenBudgetExceeded(RunStopped):
    pass


class RunTimeout(RunStopped):
    pass


class RunCancelled(RunStopped):
    pass


class GuardrailTripped(RunStopped):
    def __init__(self, message: str, guardrail: str, stage: str, info: Any = None, partial: RunResult | None = None):
        super().__init__(message, partial)
        self.guardrail = guardrail
        self.stage = stage
        self.info = info


class OutputValidationError(RunStopped):
    """The model could not produce output matching ``output_type`` after retries."""


class MCPError(OpenHarnessError):
    def __init__(self, message: str, code: int | None = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data
