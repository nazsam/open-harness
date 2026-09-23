"""A few safe, dependency-free tools that the harness and CLI enable by default."""

from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .errors import ToolError
from .tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCS = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "log",
        "log10",
        "log2",
        "exp",
        "floor",
        "ceil",
        "factorial",
        "radians",
        "degrees",
    )
}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def safe_eval(expression: str) -> float | int:
    """Evaluate arithmetic without ``eval``: numbers, + - * / // % **, and math functions."""

    def ev(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ToolError("Exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError(f"Unsupported expression: {ast.dump(node)[:80]}")

    try:
        return ev(ast.parse(expression.replace("^", "**"), mode="eval"))
    except (SyntaxError, ZeroDivisionError, ValueError, OverflowError) as e:
        raise ToolError(f"Could not evaluate '{expression}': {e}") from e


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression exactly. Use this instead of doing arithmetic in your head.

    Args:
        expression: For example "(12.5 * 4) / 3" or "sqrt(2) ** 3". Supports + - * / // % ** and
            sqrt, log, sin, cos, tan, exp, floor, ceil, abs, round, min, max, pi, e.
    """
    result = safe_eval(expression)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return str(result)


@tool
def current_time(timezone_name: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone_name: IANA time zone such as "America/Toronto" or "Europe/London".
    """
    try:
        tz = ZoneInfo(timezone_name) if timezone_name.upper() != "UTC" else timezone.utc
    except ZoneInfoNotFoundError as e:
        raise ToolError(f"Unknown time zone '{timezone_name}'") from e
    now = datetime.now(tz)
    return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z (UTC%z)")


_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)


@tool(needs_approval=True)
async def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Download a web page and return its text (HTML tags removed).

    Args:
        url: An http or https URL.
        max_chars: Maximum characters to return.
    """
    if not url.startswith(("http://", "https://")):
        raise ToolError("Only http and https URLs are allowed")
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "openharness"})
    text = resp.text
    if "html" in resp.headers.get("content-type", ""):
        text = re.sub(r"\s+", " ", _TAGS.sub(" ", text))
    return f"HTTP {resp.status_code}\n{text.strip()[:max_chars]}"


DEFAULT_TOOLS = [calculator, current_time]
ALL_TOOLS = {"calculator": calculator, "current_time": current_time, "fetch_url": fetch_url}
