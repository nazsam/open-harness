"""The ``Agent``: instructions + model + tools, and how agents compose."""

from __future__ import annotations

import dataclasses
import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .context import RunContext, RunLimits
from .guardrails import Guardrail
from .models import Model, get_model
from .tools import Tool, function_tool
from .types import Message, ModelSettings

if TYPE_CHECKING:
    from .mcp import MCPServer
    from .memory import Memory

Instructions = str | Callable[[RunContext], "str | Awaitable[str]"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower() or "agent"


@dataclass
class Handoff:
    """Lets one agent pass the conversation to another.

    The model sees a tool named ``transfer_to_<agent>``. When it calls it, the
    target agent takes over with the full history (or ``input_filter(history)``).
    """

    agent: Agent
    tool_name: str | None = None
    description: str | None = None
    input_filter: Callable[[list[Message]], list[Message]] | None = None

    @property
    def name(self) -> str:
        return self.tool_name or f"transfer_to_{_slug(self.agent.name)}"

    def tool(self) -> Tool:
        desc = self.description or (
            f"Hand the conversation to the {self.agent.name} agent."
            + (f" {self.agent.description}" if self.agent.description else "")
        )

        async def noop(_args: dict[str, Any], _ctx: RunContext) -> str:  # handled by the runner
            return f"Transferred to {self.agent.name}."

        return Tool(
            name=self.name,
            description=desc,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=noop,
            metadata={"handoff": self},
        )


@dataclass
class Agent:
    """An LLM with instructions and tools.

    Args:
        name: Shown in traces and used for handoff tool names.
        instructions: System prompt, or a function of the run context that returns one.
        model: ``"provider:model"`` string or a ``Model``. Defaults to the environment.
        tools: Tools (``@tool`` functions, plain functions, or ``Tool`` objects).
        handoffs: Agents (or ``Handoff``s) this agent may pass control to.
        output_type: Python type for structured output (dataclass, Pydantic model,
            TypedDict, list[...], etc.) or a raw JSON Schema dict.
        input_guardrails / output_guardrails: Checks run before and after the model.
        mcp_servers: MCP servers whose tools the agent can use.
        memory: A long-term memory store. Adds ``remember``/``recall`` tools.
        model_settings: Temperature, max tokens, tool choice and so on.
        limits: Default ``RunLimits`` when this agent starts a run.
        description: One line used when this agent is offered as a handoff or tool.
    """

    name: str = "assistant"
    instructions: Instructions = "You are a helpful assistant."
    model: str | Model | None = None
    tools: list[Tool | Callable[..., Any]] = field(default_factory=list)
    handoffs: list[Agent | Handoff] = field(default_factory=list)
    output_type: Any = None
    input_guardrails: list[Guardrail] = field(default_factory=list)
    output_guardrails: list[Guardrail] = field(default_factory=list)
    mcp_servers: list[MCPServer] = field(default_factory=list)
    memory: Memory | None = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    limits: RunLimits | None = None
    description: str = ""

    def __post_init__(self) -> None:
        self.tools = [t if isinstance(t, Tool) else function_tool(t) for t in self.tools]
        self._model: Model | None = None

    # ------------------------------------------------------------ helpers

    def get_model(self, override: str | Model | None = None) -> Model:
        if override is not None:
            return get_model(override)
        if self._model is None:
            self._model = get_model(self.model)
        return self._model

    async def render_instructions(self, ctx: RunContext) -> str:
        if callable(self.instructions):
            value = self.instructions(ctx)
            if inspect.isawaitable(value):
                value = await value
            return str(value)
        return self.instructions

    def handoff_objects(self) -> list[Handoff]:
        return [h if isinstance(h, Handoff) else Handoff(h) for h in self.handoffs]

    def clone(self, **changes: Any) -> Agent:
        """Copy the agent with some fields changed."""
        return dataclasses.replace(self, **changes)

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        output_extractor: Callable[[Any], Any] | None = None,
    ) -> Tool:
        """Expose this agent as a tool another agent can call (agents as workers).

        Unlike a handoff, control returns to the calling agent with the result.
        """
        agent = self

        async def handler(args: dict[str, Any], ctx: RunContext) -> Any:
            from .runner import run

            result = await run(
                agent, args["input"], deps=ctx.deps, _parent_trace=ctx.trace, cancel_token=ctx.cancel_token
            )
            ctx.usage.add(result.usage)
            return output_extractor(result.output) if output_extractor else result.output

        return Tool(
            name=name or _slug(self.name),
            description=description or self.description or f"Ask the {self.name} agent. Returns its answer.",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string", "description": "The full task or question for this agent."}},
                "required": ["input"],
                "additionalProperties": False,
            },
            handler=handler,
            metadata={"agent": self.name},
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r})"
