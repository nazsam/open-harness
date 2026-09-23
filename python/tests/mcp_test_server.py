"""Tiny MCP stdio server used by the tests. ``--mode modern`` or ``--mode legacy``."""

import json
import sys

MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "modern"
TOOLS = [
    {
        "name": "add",
        "description": "Add two integers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {"name": "fail", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
]
initialized = False


def reply(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if mid is None:
        continue  # notification
    if MODE == "modern":
        meta = params.get("_meta") or {}
        if method == "server/discover":
            reply(
                mid,
                {
                    "resultType": "complete",
                    "supportedVersions": ["2026-07-28"],
                    "serverInfo": {"name": "test", "version": "1"},
                    "capabilities": {"tools": {}},
                },
            )
            continue
        if meta.get("io.modelcontextprotocol/protocolVersion") != "2026-07-28":
            reply(mid, error={"code": -32602, "message": "missing _meta"})
            continue
    else:
        if method == "initialize":
            initialized = True
            reply(
                mid,
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "legacy-test", "version": "1"},
                },
            )
            continue
        if not initialized:
            reply(mid, error={"code": -32601, "message": f"Method not found: {method}"})
            continue
    if method == "tools/list":
        reply(mid, {"resultType": "complete", "tools": TOOLS} if MODE == "modern" else {"tools": TOOLS})
    elif method == "tools/call":
        name, args = params["name"], params.get("arguments") or {}
        if name == "add":
            reply(mid, {"content": [{"type": "text", "text": str(args["a"] + args["b"])}], "isError": False})
        else:
            reply(mid, {"content": [{"type": "text", "text": "boom"}], "isError": True})
    else:
        reply(mid, error={"code": -32601, "message": "Method not found"})
