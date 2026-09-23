import io
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from openharness import Agent, FakeModel, Harness, MCPServerHTTP, MCPServerStdio, call, run
from openharness.cli import main as cli_main
from openharness.errors import MCPError

SERVER = str(Path(__file__).with_name("mcp_test_server.py"))


@pytest.mark.parametrize("mode", ["modern", "legacy"])
async def test_mcp_stdio_both_eras(mode):
    async with MCPServerStdio(sys.executable, [SERVER, "--mode", mode], probe_timeout=2) as server:
        assert server.era == mode
        tools = await server.list_tools()
        assert [t.name for t in tools] == ["add", "fail"]
        model = FakeModel([call("add", a=20, b=22), call("fail"), "done"])
        result = await run(Agent(model=model, mcp_servers=[server]), "add")
        outs = [m for m in result.new_messages if m.role == "tool"]
        assert outs[0].content == "42" and not outs[0].is_error
        assert outs[1].is_error and "boom" in outs[1].content
        assert model.requests[0].tools[0].parameters["required"] == ["a", "b"]


def _http_server(mode):
    state = {"session": None, "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        state["requests"].append((dict(request.headers), body))
        method, params = body.get("method"), body.get("params") or {}
        if request.method == "DELETE":
            return httpx.Response(200)
        if mode == "modern":
            meta = params.get("_meta") or {}
            if request.headers.get("mcp-method") != method:
                return httpx.Response(
                    400,
                    json={
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "error": {"code": -32020, "message": "Header mismatch"},
                    },
                )
            if method == "server/discover":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "resultType": "complete",
                            "supportedVersions": ["2026-07-28"],
                            "serverInfo": {"name": "h"},
                        },
                    },
                )
            assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
        else:
            if method == "initialize":
                state["session"] = "sess-1"
                return httpx.Response(
                    200,
                    headers={"mcp-session-id": "sess-1"},
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {"protocolVersion": "2025-11-25", "capabilities": {}},
                    },
                )
            if "id" not in body:
                return httpx.Response(202)
            if request.headers.get("mcp-session-id") != "sess-1":
                return httpx.Response(400, text="Bad Request: no session")
        if method == "tools/list":
            tools = [
                {
                    "name": "echo",
                    "description": "Echo",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "region": {"type": "string", "x-mcp-header": "Region"},
                        },
                    },
                },
                {
                    "name": "bad",
                    "inputSchema": {"type": "object", "properties": {"n": {"type": "number", "x-mcp-header": "N"}}},
                },
            ]
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": tools}})
        if method == "tools/call":
            if mode == "modern":
                assert request.headers["mcp-name"] == params["name"]
            text = f"{params['arguments']['text']}|{request.headers.get('mcp-param-region', '-')}"
            sse = (
                f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'method': 'notifications/progress'})}\n\n"
                f"data: {json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': {'content': [{'type': 'text', 'text': text}]}})}\n\n"
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())
        return httpx.Response(
            404, json={"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": "not found"}}
        )

    return handler, state


@pytest.mark.parametrize("mode", ["modern", "legacy"])
async def test_mcp_http_both_eras(mode):
    handler, state = _http_server(mode)
    server = MCPServerHTTP(
        "https://mcp.example.com/mcp", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    tools = await server.list_tools()
    assert server.era == mode
    assert [t.name for t in tools] == ["echo"]  # 'bad' has an invalid x-mcp-header and is excluded
    result = await server.call_tool("echo", {"text": "hi", "region": "Café"})
    expected_region = "=?base64?Q2Fmw6k=?=" if mode == "modern" else "-"
    assert result["content"][0]["text"] == f"hi|{expected_region}"
    await server.close()


async def test_mcp_stdio_bad_command():
    server = MCPServerStdio(sys.executable, ["-c", "import sys; sys.exit(3)"], probe_timeout=1, timeout=2)
    with pytest.raises((MCPError, TimeoutError, Exception)):
        await server.list_tools()
    await server.close()


# ------------------------------------------------------------------ harness and CLI


async def test_harness_remembers_conversation_and_streams():
    h = Harness(FakeModel(["Hello Sam!", lambda r: f"{len(r.messages)} messages so far"]))
    assert await h.ask("I'm Sam") == "Hello Sam!"
    chunks = [c async for c in h.stream("again")]
    assert "".join(chunks) == "3 messages so far"
    assert h.usage.requests == 2
    assert [t.name for t in h.agent.tools] == ["calculator", "current_time"]
    await h.reset()
    assert await h.session.get_messages() == []


async def test_terminal_chat_session():
    h = Harness(FakeModel([call("calculator", expression="6*7"), "It's 42."]))
    lines = iter(["what is 6*7?", "/usage", "/tools", "/exit"])
    out = io.StringIO()
    await h.chat(input_fn=lambda _prompt: next(lines), out=out)
    text = out.getvalue()
    assert "-> calculator" in text and "<- ok: 42" in text and "It's 42." in text
    assert "'requests': 2" in text and "current_time" in text


def test_cli_run_command(capsys):
    assert cli_main(["run", "--model", "fake:demo", "hello", "there"]) == 0
    assert capsys.readouterr().out.strip() == "echo: hello there"


def test_cli_entry_point_installed():
    out = subprocess.run(
        [sys.executable, "-m", "openharness.cli", "models"], capture_output=True, text=True, check=True
    )
    assert "anthropic" in out.stdout and "ollama" in out.stdout


def test_calculator_is_safe():
    from openharness.builtin_tools import safe_eval
    from openharness.errors import ToolError

    assert safe_eval("2 ** 10 + sqrt(16)") == 1028
    with pytest.raises(ToolError):
        safe_eval("__import__('os').system('echo hi')")
