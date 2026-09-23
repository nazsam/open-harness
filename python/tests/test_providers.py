"""Provider adapters tested against recorded-style responses via httpx.MockTransport."""

import json

import httpx
import pytest

from openharness import Agent, AnthropicModel, GeminiModel, OpenAIModel, get_model, run, run_stream, tool
from openharness.errors import ConfigError, ModelError
from openharness.models.base import HTTPClient
from openharness.types import Message, ToolCall


@tool
def get_weather(city: str) -> str:
    """Get weather.

    Args:
        city: City name.
    """
    return f"18C and cloudy in {city}"


def sse(events):
    body = ""
    for ev in events:
        if isinstance(ev, tuple):
            body += f"event: {ev[0]}\ndata: {json.dumps(ev[1])}\n\n"
        elif ev == "[DONE]":
            body += "data: [DONE]\n\n"
        else:
            body += f"data: {json.dumps(ev)}\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def install(model, handler):
    model.http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return model


# ------------------------------------------------------------------ OpenAI


async def test_openai_tool_loop_request_shape():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["authorization"] == "Bearer sk-test"
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": "It is 18C in Paris."}}
                ],
                "usage": {"prompt_tokens": 70, "completion_tokens": 8},
            },
        )

    model = install(OpenAIModel("gpt-test", api_key="sk-test"), handler)
    result = await run(Agent(model=model, instructions="Be brief.", tools=[get_weather]), "Weather in Paris?")
    assert result.output == "It is 18C in Paris."
    assert result.usage.input_tokens == 120 and result.usage.output_tokens == 18
    first, second = seen
    assert first["messages"][0] == {"role": "system", "content": "Be brief."}
    assert first["tools"][0]["function"]["name"] == "get_weather"
    assert second["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
    assert second["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "18C and cloudy in Paris"}


async def test_openai_streaming_with_tool_call_chunks():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        if len(calls) == 1:
            return sse(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": ""}}
                                    ]
                                }
                            }
                        ]
                    },
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}}]},
                    {
                        "choices": [
                            {
                                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Rome"}'}}]},
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                    {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 5}},
                    "[DONE]",
                ]
            )
        return sse(
            [
                {"choices": [{"delta": {"content": "Rome: "}}]},
                {"choices": [{"delta": {"content": "18C"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ]
        )

    model = install(OpenAIModel("gpt-test", api_key="k"), handler)
    stream = run_stream(Agent(model=model, tools=[get_weather]), "Rome?")
    deltas = [d async for d in stream.text()]
    result = await stream.result()
    assert deltas == ["Rome: ", "18C"] and result.output == "Rome: 18C"
    assert result.new_messages[1].tool_calls[0].arguments == {"city": "Rome"}


async def test_openai_structured_output_strict_detection():
    from dataclasses import dataclass

    @dataclass
    class Out:
        answer: str
        note: str | None = None

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "yes"}'}}]})

    model = install(OpenAIModel("gpt-test", api_key="k"), handler)
    result = await run(Agent(model=model, output_type=Out), "?")
    assert result.output == Out("yes")
    rf = bodies[0]["response_format"]["json_schema"]
    assert rf["strict"] is False  # 'note' is optional, so strict mode would reject the schema


async def test_openai_compatible_presets(monkeypatch):
    m = get_model("ollama:llama3.2")
    assert m.base_url == "http://localhost:11434/v1" and m.provider == "ollama"
    with pytest.raises(ConfigError):
        get_model("groq:llama")  # key required
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert get_model("groq:llama").base_url.startswith("https://api.groq.com")
    body = m.build_body(
        __import__("openharness").types.ModelRequest(
            system=None,
            messages=[Message.user("hi")],
            settings=__import__("openharness").ModelSettings(max_output_tokens=100),
        ),
        stream=False,
    )
    assert body["max_tokens"] == 100  # compatible servers get max_tokens, OpenAI gets max_completion_tokens


async def test_http_retries_on_429():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"ok": True})

    client = HTTPClient("https://x", {}, provider="t", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await client.post_json("y", {}) == {"ok": True} and len(attempts) == 3


async def test_http_error_message():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    model = install(OpenAIModel("x", api_key="k", max_retries=0), handler)
    with pytest.raises(ModelError, match="bad model"):
        await run(Agent(model=model), "hi")


# ------------------------------------------------------------------ Anthropic


async def test_anthropic_tool_loop_preserves_thinking_blocks():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["x-api-key"] == "ak" and request.headers["anthropic-version"] == "2023-06-01"
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 30, "output_tokens": 9},
                    "content": [
                        {"type": "thinking", "thinking": "need weather", "signature": "sig123"},
                        {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 60, "output_tokens": 5},
                "content": [{"type": "text", "text": "Oslo is 18C."}],
            },
        )

    model = install(AnthropicModel("claude-test", api_key="ak"), handler)
    result = await run(Agent(model=model, instructions="Sys.", tools=[get_weather]), "Oslo?")
    assert result.output == "Oslo is 18C."
    first, second = seen
    assert first["system"] == "Sys." and first["max_tokens"] == 16000
    assert first["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    assistant = second["messages"][1]
    assert assistant["content"][0] == {"type": "thinking", "thinking": "need weather", "signature": "sig123"}
    assert second["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "18C and cloudy in Oslo",
    }


async def test_anthropic_streaming():
    def handler(request):
        return sse(
            [
                (
                    "message_start",
                    {"type": "message_start", "message": {"usage": {"input_tokens": 12, "output_tokens": 1}}},
                ),
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": '{"city": "L'},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": 'ima"}'},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 1}),
                (
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}},
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )

    model = install(AnthropicModel("claude-test", api_key="ak"), handler)
    from openharness.types import ModelRequest

    events = [e async for e in model.stream(ModelRequest(system=None, messages=[Message.user("hi")]))]
    assert [e.data.get("delta") for e in events if e.type == "text_delta"] == ["Hel", "lo"]
    done = events[-1].response
    assert done.message.content == "Hello"
    assert done.message.tool_calls[0].arguments == {"city": "Lima"}
    assert done.usage.input_tokens == 12 and done.usage.output_tokens == 20


def test_anthropic_merges_consecutive_tool_results_and_structured_output():
    from openharness.types import ModelRequest, OutputSchema

    model = AnthropicModel("c", api_key="k")
    msgs = [
        Message.user("q"),
        Message.assistant("", [ToolCall("a", "x", {}), ToolCall("b", "y", {})]),
        Message.tool("a", "x", "1"),
        Message.tool("b", "y", "2", is_error=True),
        Message.user("more"),
    ]
    body = model.build_body(
        ModelRequest(system="s", messages=msgs, output_schema=OutputSchema("o", {"type": "object"})), stream=False
    )
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    last = body["messages"][2]["content"]
    assert [b["type"] for b in last] == ["tool_result", "tool_result", "text"] and last[1]["is_error"] is True
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}


# ------------------------------------------------------------------ Gemini


async def test_gemini_tool_loop_keeps_thought_signature():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        assert request.headers["x-goog-api-key"] == "gk"
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {"name": "get_weather", "args": {"city": "Lagos"}},
                                        "thoughtSignature": "abc",
                                    }
                                ],
                            },
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 4, "thoughtsTokenCount": 6},
                },
            )
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "Lagos: 18C"}]}}]})

    model = install(GeminiModel("gemini-test", api_key="gk"), handler)
    result = await run(Agent(model=model, instructions="Sys.", tools=[get_weather]), "Lagos?")
    assert result.output == "Lagos: 18C"
    (path1, first), (_, second) = seen
    assert path1.endswith("/models/gemini-test:generateContent")
    assert first["systemInstruction"] == {"parts": [{"text": "Sys."}]}
    assert first["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["required"] == ["city"]
    assert second["contents"][1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Lagos"}}, "thoughtSignature": "abc"}],
    }
    fr = second["contents"][2]["parts"][0]["functionResponse"]
    assert fr == {"name": "get_weather", "response": {"result": "18C and cloudy in Lagos"}}
    assert result.usage.output_tokens == 10


async def test_gemini_streaming():
    def handler(request):
        assert request.url.params["alt"] == "sse"
        return sse(
            [
                {"candidates": [{"content": {"parts": [{"text": "Hi "}]}}]},
                {
                    "candidates": [{"content": {"parts": [{"text": "there"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
                },
            ]
        )

    model = install(GeminiModel("gemini-test", api_key="gk"), handler)
    stream = run_stream(Agent(model=model), "hi")
    assert [d async for d in stream.text()] == ["Hi ", "there"]
    assert (await stream.result()).output == "Hi there"


# ------------------------------------------------------------------ switching providers


def test_history_moves_between_providers():
    """A conversation started on Anthropic continues on OpenAI and Gemini."""
    from openharness.types import ModelRequest

    assistant = Message.assistant("Checking.", [ToolCall("toolu_1", "get_weather", {"city": "Oslo"})])
    assistant.raw = {"provider": "anthropic", "data": [{"type": "thinking", "thinking": "x", "signature": "s"}]}
    msgs = [Message.user("Oslo?"), assistant, Message.tool("toolu_1", "get_weather", "18C")]
    req = ModelRequest(system="s", messages=msgs)
    oa = OpenAIModel("gpt", api_key="k").build_body(req, stream=False)
    assert oa["messages"][2]["tool_calls"][0]["id"] == "toolu_1" and oa["messages"][3]["role"] == "tool"
    gm = GeminiModel("g", api_key="k").build_body(req)
    assert gm["contents"][1]["parts"][1]["functionCall"] == {
        "name": "get_weather",
        "args": {"city": "Oslo"},
        "id": "toolu_1",
    }
    assert "thinking" not in json.dumps(gm)  # other providers' reasoning is not forwarded


def test_get_model_inference(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert get_model("claude-sonnet-5").provider == "anthropic"
    monkeypatch.delenv("OPENHARNESS_MODEL", raising=False)
    assert get_model().model == "claude-sonnet-5"
    with pytest.raises(ConfigError):
        get_model("mystery-model")
