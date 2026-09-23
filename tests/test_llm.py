import json

import httpx
import pytest

from twagent.llm import LLMError, OpenAICompatClient, ScriptedLLM


def _client(handler, **kw):
    return OpenAICompatClient(
        "http://llm.test/v1/", "qwen", transport=httpx.MockTransport(handler), **kw
    )


def test_request_body_is_greedy_and_has_no_tools():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})

    out = _client(handler).complete([{"role": "user", "content": "嗨"}])
    assert out == "好"
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    body = seen["body"]
    assert body["temperature"] == 0.0
    assert body["stream"] is False
    assert "tools" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "嗨"}]


def test_extra_body_is_merged():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    _client(handler, extra_body={"lora": [{"id": 0, "scale": 0.0}]}).complete([])
    assert seen["lora"] == [{"id": 0, "scale": 0.0}]


def test_null_content_becomes_empty_string():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    assert _client(handler).complete([]) == ""


def test_http_error_raises_llm_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError, match="failed"):
        _client(handler).complete([])


def test_bad_shape_raises_llm_error():
    def handler(request):
        return httpx.Response(200, json={"oops": 1})

    with pytest.raises(LLMError, match="unexpected"):
        _client(handler).complete([])


def test_scripted_llm_replays_and_records():
    llm = ScriptedLLM(["a", "b"])
    assert llm.complete([{"role": "user", "content": "1"}]) == "a"
    assert llm.complete([]) == "b"
    assert llm.calls[0] == [{"role": "user", "content": "1"}]
    with pytest.raises(AssertionError, match="ran out"):
        llm.complete([])
