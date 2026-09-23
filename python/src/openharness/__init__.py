"""open-harness: a self-hosted SDK for building AI agents.

from openharness import Agent, run_sync, tool

@tool
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b

agent = Agent(instructions="Be brief.", model="anthropic:claude-sonnet-5", tools=[add])
print(run_sync(agent, "What is 2 + 40?").output)
"""

from ._version import __version__
from .agent import Agent, Handoff
from .builtin_tools import calculator, current_time, fetch_url
from .context import CancelToken, RunContext, RunLimits
from .errors import (
    ConfigError,
    GuardrailTripped,
    MaxToolCallsExceeded,
    MaxTurnsExceeded,
    MCPError,
    ModelError,
    OpenHarnessError,
    OutputValidationError,
    RunCancelled,
    RunStopped,
    RunTimeout,
    TokenBudgetExceeded,
    ToolError,
)
from .guardrails import (
    Guardrail,
    GuardrailResult,
    blocked_patterns,
    input_guardrail,
    llm_guardrail,
    max_length,
    output_guardrail,
    pii,
)
from .harness import Harness
from .mcp import MCPServer, MCPServerHTTP, MCPServerStdio
from .memory import FileMemory, InMemoryMemory, Memory, MemoryItem
from .models import (
    AnthropicModel,
    FakeModel,
    GeminiModel,
    Model,
    ModelEvent,
    OpenAIModel,
    call,
    get_model,
)
from .result import RunResult, RunStream
from .runner import run, run_stream, run_sync
from .sessions import FileSession, InMemorySession, Session, SQLiteSession
from .tools import Tool, function_tool, tool
from .tracing import (
    ConsoleProcessor,
    JSONLProcessor,
    MemoryProcessor,
    OpenTelemetryProcessor,
    Span,
    Trace,
    add_trace_processor,
    clear_trace_processors,
)
from .types import Message, ModelSettings, StreamEvent, ToolCall, Usage

__all__ = [
    "__version__",
    # core
    "Agent",
    "Handoff",
    "run",
    "run_sync",
    "run_stream",
    "RunResult",
    "RunStream",
    "Harness",
    "tool",
    "function_tool",
    "Tool",
    "RunContext",
    "RunLimits",
    "CancelToken",
    "Message",
    "ToolCall",
    "Usage",
    "ModelSettings",
    "StreamEvent",
    # models
    "Model",
    "ModelEvent",
    "get_model",
    "OpenAIModel",
    "AnthropicModel",
    "GeminiModel",
    "FakeModel",
    "call",
    # sessions and memory
    "Session",
    "InMemorySession",
    "FileSession",
    "SQLiteSession",
    "Memory",
    "MemoryItem",
    "InMemoryMemory",
    "FileMemory",
    # guardrails
    "Guardrail",
    "GuardrailResult",
    "input_guardrail",
    "output_guardrail",
    "max_length",
    "blocked_patterns",
    "pii",
    "llm_guardrail",
    # tracing
    "Trace",
    "Span",
    "ConsoleProcessor",
    "JSONLProcessor",
    "MemoryProcessor",
    "OpenTelemetryProcessor",
    "add_trace_processor",
    "clear_trace_processors",
    # mcp
    "MCPServer",
    "MCPServerStdio",
    "MCPServerHTTP",
    # tools
    "calculator",
    "current_time",
    "fetch_url",
    # errors
    "OpenHarnessError",
    "ModelError",
    "ConfigError",
    "ToolError",
    "RunStopped",
    "MaxTurnsExceeded",
    "MaxToolCallsExceeded",
    "TokenBudgetExceeded",
    "RunTimeout",
    "RunCancelled",
    "GuardrailTripped",
    "OutputValidationError",
    "MCPError",
]
