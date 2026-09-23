"""Tools: plain Python functions the model can call.

    from openharness import tool

    @tool
    def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
        '''Get the current weather for a city.

        Args:
            city: City name, e.g. "Toronto".
            unit: Temperature unit.
        '''
        ...

The JSON schema is built from type hints and the docstring. A parameter
annotated with ``RunContext`` receives the live run context instead of a
model-supplied value.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_type_hints

from .errors import ToolError
from .schema import SchemaValidationError, coerce, type_to_schema, validate
from .types import ToolSpec

if TYPE_CHECKING:
    from .context import RunContext

ToolHandler = Callable[..., Any]


@dataclass
class Tool:
    """A tool the model can call.

    Most users create tools with the ``@tool`` decorator. Build one directly
    to wrap an existing schema (this is how MCP tools are exposed).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], RunContext], Awaitable[Any]]
    needs_approval: bool | Callable[[dict[str, Any]], bool] = False
    timeout: float | None = None
    strict: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    def requires_approval(self, args: dict[str, Any]) -> bool:
        if callable(self.needs_approval):
            return bool(self.needs_approval(args))
        return bool(self.needs_approval)

    async def invoke(self, args: dict[str, Any], ctx: RunContext) -> Any:
        if self.parameters:
            try:
                validate(args, self.parameters)
            except SchemaValidationError as e:
                raise ToolError(f"Invalid arguments for {self.name}: {e}") from e
        coro = self.handler(args, ctx)
        if self.timeout:
            try:
                return await asyncio.wait_for(coro, self.timeout)
            except asyncio.TimeoutError as e:
                raise ToolError(f"Tool {self.name} timed out after {self.timeout}s") from e
        return await coro


def _context_param(fn: Callable[..., Any], hints: dict[str, Any]) -> str | None:
    from .context import RunContext

    for name, hint in hints.items():
        if hint is RunContext or getattr(hint, "__origin__", None) is RunContext:
            return name
    return None


_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters|Params)\s*:\s*$", re.IGNORECASE)
_SECTION = re.compile(r"^\s*(Returns?|Raises?|Yields?|Examples?|Notes?)\s*:\s*$", re.IGNORECASE)
_PARAM_LINE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    params: dict[str, str] = {}
    mode, current = "summary", None
    for line in lines:
        if _ARGS_HEADER.match(line):
            mode = "args"
            continue
        if _SECTION.match(line):
            mode = "other"
            continue
        if mode == "summary":
            summary.append(line)
        elif mode == "args":
            m = _PARAM_LINE.match(line)
            if m and (len(line) - len(line.lstrip())) <= 4:
                current = m.group(1)
                params[current] = m.group(2).strip()
            elif current and line.strip():
                params[current] += " " + line.strip()
    return "\n".join(summary).strip(), params


def function_tool(
    fn: ToolHandler,
    *,
    name: str | None = None,
    description: str | None = None,
    needs_approval: bool | Callable[[dict[str, Any]], bool] = False,
    timeout: float | None = None,
) -> Tool:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    ctx_name = _context_param(fn, hints)
    summary, param_docs = parse_docstring(fn.__doc__)
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, p in sig.parameters.items():
        if pname == ctx_name or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        hint = hints.get(pname, p.annotation if p.annotation is not p.empty else str)
        prop = type_to_schema(hint)
        if pname in param_docs and "description" not in prop:
            prop["description"] = param_docs[pname]
        if p.default is p.empty:
            required.append(pname)
        elif p.default is not None and isinstance(p.default, (str, int, float, bool)):
            prop.setdefault("default", p.default)
        props[pname] = prop
    parameters: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        parameters["required"] = required

    is_async = inspect.iscoroutinefunction(fn)

    async def handler(args: dict[str, Any], ctx: RunContext) -> Any:
        kwargs = {k: coerce(v, hints.get(k, Any)) for k, v in args.items() if k in sig.parameters}
        if ctx_name:
            kwargs[ctx_name] = ctx
        if is_async:
            return await fn(**kwargs)
        return await asyncio.to_thread(fn, **kwargs)

    return Tool(
        name=name or fn.__name__,
        description=description or summary or f"Call {fn.__name__}",
        parameters=parameters,
        handler=handler,
        needs_approval=needs_approval,
        timeout=timeout,
        metadata={"source": "function"},
    )


def tool(
    fn: ToolHandler | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    needs_approval: bool | Callable[[dict[str, Any]], bool] = False,
    timeout: float | None = None,
) -> Any:
    """Decorator that turns a function into a ``Tool``. Works with or without arguments."""

    def wrap(f: ToolHandler) -> Tool:
        return function_tool(f, name=name, description=description, needs_approval=needs_approval, timeout=timeout)

    return wrap(fn) if fn is not None else wrap
