import json
from datetime import date

import pytest

from twagent.agent import FAILURE_MESSAGE, MAX_STEPS_MESSAGE, run_agent
from twagent.config import AgentConfig
from twagent.llm import LLMError, ScriptedLLM
from twagent.prompting import build_system_prompt
from twagent.tools.mock import MockTools

SYSTEM = build_system_prompt(date(2026, 9, 23))
USER = [{"role": "user", "content": "明天台中會下雨嗎"}]
CFG = AgentConfig(max_steps=5, parse_retries=1, schema_retries=2)


def call(name, **args):
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        + "\n</tool_call>"
    )


WEATHER = call("get_weather", city="臺中市", date="2026-09-24")


class FailingLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        raise LLMError("connection refused")


def run(llm, tools=None, fallback=None, cfg=CFG):
    return run_agent(
        USER,
        llm=llm,
        tools=tools or MockTools(),
        system_prompt=SYSTEM,
        config=cfg,
        fallback=fallback,
    )


def test_direct_answer():
    r = run(ScriptedLLM(["你好！"]))
    assert r.stopped_reason == "answer" and r.final_text == "你好！"
    assert r.tool_calls == []
    assert r.messages[0] == {"role": "system", "content": SYSTEM}


def test_single_tool_round_trip():
    llm = ScriptedLLM([WEATHER, "明天臺中多雲，降雨機率 20%。"])
    r = run(llm)
    assert r.stopped_reason == "answer"
    assert [c.name for c in r.tool_calls] == ["get_weather"]
    # Second LLM call sees: system, user, assistant(tool_call), tool(result)
    second = llm.calls[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["content"] == WEATHER
    assert json.loads(second[3]["content"])["city"] == "臺中市"


def test_parallel_calls_each_get_a_tool_message():
    both = WEATHER + "\n" + call("get_weather", city="高雄市", date="2026-09-24")
    llm = ScriptedLLM([both, "兩地都晴。"])
    r = run(llm)
    assert [c.arguments["city"] for c in r.tool_calls] == ["臺中市", "高雄市"]
    assert [m["role"] for m in llm.calls[1]][-2:] == ["tool", "tool"]


def test_sequential_calls_update_calendar_state():
    tools = MockTools()
    llm = ScriptedLLM(
        [
            call(
                "search_thsr",
                origin="臺北",
                destination="臺南",
                date="2026-09-24",
                depart_after="07:00",
            ),
            call(
                "calendar_add", title="高鐵去臺南", start="2026-09-24T07:30", end="2026-09-24T09:30"
            ),
            "已幫你排進行事曆。",
        ]
    )
    r = run(llm, tools=tools)
    assert r.stopped_reason == "answer"
    assert [e["title"] for e in tools.events()] == ["高鐵去臺南"]


def test_parse_error_then_successful_retry():
    llm = ScriptedLLM(
        ['<tool_call>{"name": "get_weather", "arguments": {</tool_call>', WEATHER, "多雲"]
    )
    r = run(llm)
    assert r.stopped_reason == "answer"
    assert [e.type for e in r.trace].count("parse_error") == 1
    retry_prompt = llm.calls[1][-1]
    assert retry_prompt["role"] == "user" and "格式錯誤" in retry_prompt["content"]


def test_parse_failure_without_fallback_gives_up():
    bad = "<tool_call>oops</tool_call>"
    r = run(ScriptedLLM([bad, bad]))
    assert r.stopped_reason == "parse_failure" and r.final_text == FAILURE_MESSAGE


def test_parse_failure_escalates_to_fallback_without_garbage():
    bad = "<tool_call>oops</tool_call>"
    fallback = ScriptedLLM([WEATHER, "多雲"])
    r = run(ScriptedLLM([bad, bad]), fallback=fallback)
    assert r.stopped_reason == "answer" and r.used_fallback
    # Fallback's first view is the clean conversation: no bad output, no retry prompt.
    assert [m["role"] for m in fallback.calls[0]] == ["system", "user"]


def test_schema_error_is_fed_back_and_corrected():
    wrong = call("get_weather", city="台中", date="明天")
    llm = ScriptedLLM([wrong, WEATHER, "多雲"])
    r = run(llm)
    assert r.stopped_reason == "answer"
    assert [c.arguments["city"] for c in r.tool_calls] == ["臺中市"]  # invalid call not executed
    feedback = json.loads(llm.calls[1][-1]["content"])
    assert feedback["error"] == "參數驗證失敗" and feedback["details"]


def test_schema_retries_exhausted():
    wrong = call("get_weather", city="台中", date="明天")
    r = run(ScriptedLLM([wrong] * 3))
    assert r.stopped_reason == "schema_failure" and r.final_text == FAILURE_MESSAGE


def test_tool_error_is_reported_not_replaced():
    tools = MockTools(failures={"get_weather": "連線逾時"})
    llm = ScriptedLLM([WEATHER, "抱歉，氣象服務暫時連不上。"])
    r = run(llm, tools=tools)
    assert r.stopped_reason == "answer"
    assert json.loads(llm.calls[1][-1]["content"]) == {"error": "連線逾時"}
    assert [e.type for e in r.trace] == ["llm_output", "tool_error", "llm_output"]


def test_llm_error_without_fallback():
    r = run(FailingLLM())
    assert r.stopped_reason == "llm_error" and r.final_text == FAILURE_MESSAGE


def test_llm_error_with_fallback():
    r = run(FailingLLM(), fallback=ScriptedLLM(["你好"]))
    assert r.stopped_reason == "answer" and r.used_fallback


def test_max_steps():
    r = run(ScriptedLLM([WEATHER] * 5))
    assert r.stopped_reason == "max_steps" and r.final_text == MAX_STEPS_MESSAGE
    assert len(r.tool_calls) == 5


@pytest.mark.parametrize("steps", [1, 2])
def test_max_steps_counts_every_llm_call(steps):
    llm = ScriptedLLM([WEATHER] * steps)
    run(llm, cfg=AgentConfig(max_steps=steps))
    assert len(llm.calls) == steps


def test_to_dict_is_json_serialisable():
    r = run(ScriptedLLM([WEATHER, "多雲"]))
    d = r.to_dict()
    json.dumps(d, ensure_ascii=False)
    assert d["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-09-24"}}
    ]
