import json
from datetime import date

import pytest

from twagent.data.scenarios import Scenario
from twagent.data.synth import run_trajectory
from twagent.data.validate import reason_key, validate_record, wording_problem
from twagent.llm import ScriptedLLM

TODAY = date(2026, 3, 4)  # Wednesday; 明天 = 2026-03-05


def call(name, **args):
    body = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


def record(category, tools, replies, message="明天台中會下雨嗎", date_expr="明天", **kw):
    s = Scenario("s1", category, tuple(tools), "口語隨興", TODAY, date_expr, **kw)
    return run_trajectory(s, message, ScriptedLLM(replies))


W = call("get_weather", city="臺中市", date="2026-03-05")
W2 = call("get_weather", city="高雄市", date="2026-03-05")


def test_clean_single_is_kept():
    assert validate_record(record("single", ["get_weather"], [W, "明天臺中多雲。"])) == []


def test_wrong_date_is_rejected():
    bad = call("get_weather", city="臺中市", date="2026-03-06")
    problems = validate_record(record("single", ["get_weather"], [bad, "多雲"]))
    assert problems == ["date: 明天 should be 2026-03-05, got 2026-03-06"]


def test_wrong_tool_is_rejected():
    thsr = call("search_thsr", origin="臺北", destination="臺中", date="2026-03-05")
    problems = validate_record(record("single", ["get_weather"], [thsr, "好"]))
    assert problems and problems[0].startswith("single:")


def test_parallel_needs_one_turn_with_all_calls():
    ok = record("parallel", ["get_weather", "get_weather"], [W + "\n" + W2, "兩地都晴。"])
    assert validate_record(ok) == []
    split = record("parallel", ["get_weather", "get_weather"], [W, W2, "兩地都晴。"])
    assert validate_record(split)[0].startswith("parallel:")


def test_sequential_needs_two_turns_in_order():
    thsr = call("search_thsr", origin="臺北", destination="臺南", date="2026-03-05")
    cal = call("calendar_add", title="高鐵去臺南", start="2026-03-05T08:00", end="2026-03-05T10:00")
    ok = record("sequential", ["search_thsr", "calendar_add"], [thsr, cal, "排好了。"])
    assert validate_record(ok) == []
    same_turn = record(
        "sequential", ["search_thsr", "calendar_add"], [thsr + "\n" + cal, "排好了。"]
    )
    assert validate_record(same_turn)[0].startswith("sequential:")


def test_wrong_date_on_second_call_is_rejected():
    thsr = call("search_thsr", origin="臺北", destination="臺南", date="2026-03-05")
    bad_cal = call(
        "calendar_add", title="高鐵去臺南", start="2026-03-06T08:00", end="2026-03-06T10:00"
    )
    rec = record("sequential", ["search_thsr", "calendar_add"], [thsr, bad_cal, "排好了。"])
    assert validate_record(rec) == ["date: 明天 should be 2026-03-05, got 2026-03-06"]


def test_clarify_must_ask_and_not_call():
    assert (
        validate_record(
            record("clarify", ["search_thsr"], ["請問要從哪一站出發？"], date_expr=None)
        )
        == []
    )
    assert validate_record(record("clarify", ["search_thsr"], ["好的。"], date_expr=None)) == [
        "clarify: reply is not a question"
    ]
    called = record("clarify", ["get_weather"], [W, "多雲"], date_expr=None)
    assert validate_record(called)[0].startswith("clarify: expected no calls")


def test_no_tool():
    assert validate_record(record("no_tool", [], ["你好！"], "你好", None, kind="閒聊")) == []


def test_tool_error_must_actually_fail():
    yb = call("find_youbike", city="臺北市", near="臺大")
    ok = record(
        "tool_error",
        ["find_youbike"],
        [yb, "抱歉，暫時查不到。"],
        "台大附近有車嗎",
        None,
        failure="連線逾時",
    )
    assert validate_record(ok) == []


def test_teacher_error_rejects_even_if_it_recovered():
    rec = record("single", ["get_weather"], ["<tool_call>oops</tool_call>"])
    assert validate_record(rec) == ["stopped: parse_failure"]


def test_simplified_and_mainland_wording():
    assert validate_record(record("single", ["get_weather"], [W, "明天台中多云。"]))[-1].startswith(
        "wording:"
    )
    rec = record("no_tool", [], ["這個視頻很好看"], "推薦影片", None, kind="閒聊")
    # Snippet: up to 4 characters either side of the first difference.
    assert validate_record(rec) == ["wording: 這個視頻很好 -> 這個影片很好"]


def test_tool_name_leak():
    rec = record("single", ["get_weather"], [W, "多雲"], message="用 get_weather 查明天台中")
    assert "tool name leaked into user message" in validate_record(rec)


@pytest.mark.parametrize(
    "text", ["台北明天會下雨嗎", "臺北的捷運站", "幫你排進行事曆了", "網路很慢", "這支影片"]
)
def test_wording_accepts_taiwanese(text):
    assert wording_problem(text) is None


def test_reason_key():
    assert reason_key("date: 明天 should be x, got y") == "date"
    assert reason_key("empty final reply") == "empty final reply"
