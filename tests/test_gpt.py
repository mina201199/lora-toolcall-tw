from datetime import date
from types import SimpleNamespace

import httpx
import openai
import pytest

from twagent.config import ModelPrice
from twagent.gpt import GPTClient, Usage, to_openai_messages
from twagent.llm import LLMError
from twagent.prompting import build_persona, build_system_prompt

SYSTEM = build_system_prompt(date(2026, 9, 23))
BREAKPOINT = {"mode": "explicit"}


def response(text="好", finish_reason="stop", refusal=None, prompt=10, cached=0, written=0):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text, refusal=refusal),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=written),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


class FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def client_with(result, model="gpt-6-sol", **kwargs):
    completions = FakeCompletions(result)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return GPTClient(model, client=fake, **kwargs), completions


def test_system_prompt_is_split_tools_first_with_breakpoint():
    msgs = to_openai_messages([{"role": "system", "content": SYSTEM}])
    assert [m["role"] for m in msgs] == ["system", "system"]
    [tools_part] = msgs[0]["content"]
    assert tools_part["type"] == "text"
    assert tools_part["text"].startswith("# Tools\n\nYou may call one or more functions")
    assert tools_part["prompt_cache_breakpoint"] == BREAKPOINT
    assert msgs[1]["content"] == build_persona(date(2026, 9, 23))
    # Nothing lost or added by the split.
    assert msgs[1]["content"] + "\n\n" + tools_part["text"] == SYSTEM


def test_system_without_tools_is_one_message():
    assert to_openai_messages([{"role": "system", "content": "hi"}]) == [
        {"role": "system", "content": "hi"}
    ]


def test_tool_results_become_merged_user_turns():
    msgs = to_openai_messages(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "台北跟高雄天氣"},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            {"role": "tool", "content": '{"city": "臺北市"}'},
            {"role": "tool", "content": '{"city": "高雄市"}'},
        ]
    )
    assert [m["role"] for m in msgs] == ["system", "system", "user", "assistant", "user"]
    assert msgs[4]["content"] == (
        '<tool_response>\n{"city": "臺北市"}\n</tool_response>\n'
        '<tool_response>\n{"city": "高雄市"}\n</tool_response>'
    )


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="unknown role"):
        to_openai_messages([{"role": "developer", "content": "x"}])


def test_complete_sends_reasoning_effort_and_accumulates_usage():
    client, fake = client_with(response("你好", prompt=1500, cached=1200, written=250))
    assert (
        client.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "嗨"}])
        == "你好"
    )
    sent = fake.calls[0]
    assert sent["model"] == "gpt-6-sol"
    assert sent["reasoning_effort"] == "none"
    assert sent["max_completion_tokens"] == 1024
    assert "tools" not in sent and "temperature" not in sent and "max_tokens" not in sent
    u = client.usage
    assert (u.requests, u.input_tokens, u.cached_tokens, u.cache_write_tokens) == (1, 50, 1200, 250)


def test_reasoning_effort_omitted_when_empty():
    client, fake = client_with(response(), model="gpt-4.1", reasoning_effort="")
    client.complete([{"role": "user", "content": "嗨"}])
    assert "reasoning_effort" not in fake.calls[0]


@pytest.mark.parametrize(
    ("finish_reason", "refusal"),
    [("length", None), ("content_filter", None), ("stop", "I can't help with that.")],
)
def test_bad_responses_raise(finish_reason, refusal):
    client, _ = client_with(response(finish_reason=finish_reason, refusal=refusal))
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "嗨"}])
    assert client.usage.requests == 1  # still billed, still counted


def test_api_error_becomes_llm_error():
    err = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    client, _ = client_with(err)
    with pytest.raises(LLMError, match="gpt-6-sol"):
        client.complete([{"role": "user", "content": "嗨"}])


def test_usage_cost():
    u = Usage(
        requests=1,
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_tokens=2_000_000,
        cache_write_tokens=400_000,
    )
    price = ModelPrice(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
    assert u.cost_usd(price) == pytest.approx(2.0 + 1.0 + 0.4 + 1.0)
