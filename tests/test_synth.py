import json
from datetime import date

from twagent.data.scenarios import Scenario
from twagent.data.synth import (
    MAX_CONSECUTIVE_LLM_ERRORS,
    build_query_prompt,
    load_records,
    parse_query_response,
    run_trajectory,
    synthesize,
)
from twagent.llm import LLMError, ScriptedLLM

TODAY = date(2026, 3, 4)  # Wednesday
WEATHER = Scenario("s00001", "single", ("get_weather",), "口語隨興", TODAY, "明天")
CLARIFY = Scenario("s00002", "clarify", ("search_thsr",), "非常簡短", TODAY, "後天")
CHAT = Scenario("s00003", "no_tool", (), "客氣正式", TODAY, kind="閒聊")
BROKEN = Scenario("s00004", "tool_error", ("find_youbike",), "口語隨興", TODAY, failure="連線逾時")

WEATHER_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-03-05"}}\n</tool_call>'


def queries(*pairs):
    return json.dumps([{"id": i, "message": m} for i, m in pairs], ensure_ascii=False)


def test_prompt_mentions_each_scenario_and_its_constraints():
    prompt = build_query_prompt([WEATHER, CLARIFY, CHAT])
    for s in (WEATHER, CLARIFY, CHAT):
        assert f"id：{s.id}" in prompt
    assert "必須使用「明天」" in prompt
    assert "2026-03-04（星期三）" in prompt
    assert "get_weather（查詢臺灣某縣市某一天的天氣預報。）" in prompt
    assert "類型：閒聊" in prompt
    assert "刻意漏掉" in prompt


def test_parse_query_response_tolerates_prose_and_drops_bad_items():
    raw = "好的：\n" + json.dumps(
        [
            {"id": "s00001", "message": " 明天台中會下雨嗎 "},
            {"id": "s99999", "message": "unknown id"},
            {"id": "s00002", "message": ""},
            "not a dict",
        ],
        ensure_ascii=False,
    )
    assert parse_query_response(raw, [WEATHER, CLARIFY]) == {"s00001": "明天台中會下雨嗎"}


def test_parse_query_response_garbage():
    assert parse_query_response("no json here", [WEATHER]) == {}
    assert parse_query_response("[not json]", [WEATHER]) == {}


def test_trajectory_uses_scenario_date_and_mock_results():
    teacher = ScriptedLLM([WEATHER_CALL, "明天臺中多雲。"])
    record = run_trajectory(WEATHER, "明天台中會下雨嗎", teacher)
    assert "今天是 2026-03-04（星期三）" in teacher.calls[0][0]["content"]
    assert record["result"]["stopped_reason"] == "answer"
    assert [m["role"] for m in record["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert json.loads(record["messages"][3]["content"])["city"] == "臺中市"
    assert record["scenario"]["id"] == "s00001"


def test_tool_error_scenario_makes_the_first_tool_fail():
    call = '<tool_call>\n{"name": "find_youbike", "arguments": {"city": "臺北市", "near": "臺大"}}\n</tool_call>'
    record = run_trajectory(
        BROKEN, "台大附近有 YouBike 嗎", ScriptedLLM([call, "抱歉，查詢逾時。"])
    )
    assert json.loads(record["messages"][3]["content"]) == {"error": "連線逾時"}


def test_teacher_gets_no_retries():
    record = run_trajectory(
        WEATHER, "明天台中會下雨嗎", ScriptedLLM(["<tool_call>oops</tool_call>"])
    )
    assert record["result"]["stopped_reason"] == "parse_failure"


def test_synthesize_writes_resumes_and_skips_missing(tmp_path):
    out = tmp_path / "raw.jsonl"
    query_llm = ScriptedLLM([queries(("s00001", "明天台中會下雨嗎"), ("s00003", "你好呀"))])
    teacher = ScriptedLLM([WEATHER_CALL, "明天臺中多雲。", "你好！有什麼可以幫你？"])
    report = synthesize(
        [WEATHER, CLARIFY, CHAT],
        query_llm=query_llm,
        teacher=teacher,
        out_path=out,
        batch_size=10,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert (report.written, report.missing_queries, report.stopped) == (2, 1, "complete")
    assert [r["scenario"]["id"] for r in load_records(out)] == ["s00001", "s00003"]

    # Second run only asks for what is missing.
    query_llm2 = ScriptedLLM([queries(("s00002", "幫我查高鐵"))])
    report2 = synthesize(
        [WEATHER, CLARIFY, CHAT],
        query_llm=query_llm2,
        teacher=ScriptedLLM(["請問要從哪一站出發？"]),
        out_path=out,
        batch_size=10,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert (report2.already_done, report2.written) == (2, 1)
    assert "s00001" not in query_llm2.calls[0][0]["content"]
    assert len(load_records(out)) == 3


def test_synthesize_stops_at_budget(tmp_path):
    report = synthesize(
        [WEATHER],
        query_llm=ScriptedLLM([]),
        teacher=ScriptedLLM([]),
        out_path=tmp_path / "raw.jsonl",
        batch_size=10,
        spent_usd=lambda: 5.0,
        budget_usd=5.0,
    )
    assert report.stopped == "budget" and report.written == 0


class FailingLLM:
    def complete(self, messages):
        raise LLMError("down")


def test_query_failure_counts_as_missing(tmp_path):
    # Two scenarios in one batch are one failed call: not enough to stop the run.
    report = synthesize(
        [WEATHER, CHAT],
        query_llm=FailingLLM(),
        teacher=ScriptedLLM([]),
        out_path=tmp_path / "raw.jsonl",
        batch_size=10,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert (report.written, report.missing_queries, report.stopped) == (0, 2, "complete")


def test_persistent_query_failures_stop_the_run(tmp_path):
    report = synthesize(
        [WEATHER, CLARIFY, CHAT, BROKEN],
        query_llm=FailingLLM(),
        teacher=ScriptedLLM([]),
        out_path=tmp_path / "raw.jsonl",
        batch_size=1,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert report.stopped == "llm_error"
    assert report.missing_queries == MAX_CONSECUTIVE_LLM_ERRORS == 3


def all_queries(*scenarios):
    return ScriptedLLM([queries(*[(s.id, f"{s.id} 明天台中會下雨嗎") for s in scenarios])])


def test_teacher_llm_errors_are_not_recorded_and_stop_the_run(tmp_path):
    out = tmp_path / "raw.jsonl"
    batch = [WEATHER, CLARIFY, CHAT, BROKEN]
    report = synthesize(
        batch,
        query_llm=all_queries(*batch),
        teacher=FailingLLM(),
        out_path=out,
        batch_size=10,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert (report.written, report.llm_errors, report.stopped) == (0, 3, "llm_error")
    assert load_records(out) == []


class FlakyLLM:
    """Raises LLMError on the first call, then plays `replies`."""

    def __init__(self, replies):
        self.inner, self.failed = ScriptedLLM(replies), False

    def complete(self, messages):
        if not self.failed:
            self.failed = True
            raise LLMError("transient")
        return self.inner.complete(messages)


def test_transient_teacher_failure_is_skipped_and_retried_later(tmp_path):
    out = tmp_path / "raw.jsonl"
    report = synthesize(
        [CHAT, WEATHER],
        query_llm=all_queries(CHAT, WEATHER),
        teacher=FlakyLLM([WEATHER_CALL, "明天臺中多雲。"]),
        out_path=out,
        batch_size=10,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert (report.written, report.llm_errors, report.stopped) == (1, 1, "complete")
    assert [r["scenario"]["id"] for r in load_records(out)] == ["s00001"]
    # The failed scenario is still undone, so a rerun asks for it again.
    report2 = synthesize(
        [CHAT, WEATHER],
        query_llm=all_queries(CHAT),
        teacher=ScriptedLLM(["你好！"]),
        out_path=out,
        batch_size=10,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert (report2.already_done, report2.written) == (1, 1)
    assert [r["scenario"]["id"] for r in load_records(out)] == ["s00001", "s00003"]


def test_truncated_last_line_is_repaired_before_resuming(tmp_path):
    # A hard kill mid-write leaves a partial last line behind.
    out = tmp_path / "raw.jsonl"
    full = run_trajectory(
        WEATHER, "明天台中會下雨嗎", ScriptedLLM([WEATHER_CALL, "明天臺中多雲。"])
    )
    partial = '{"scenario": {"id": "s00003", "category": "no_'
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(full, ensure_ascii=False) + "\n" + partial)
    report = synthesize(
        [WEATHER, CHAT],
        query_llm=ScriptedLLM([queries(("s00003", "你好呀"))]),
        teacher=ScriptedLLM(["你好！有什麼可以幫你？"]),
        out_path=out,
        batch_size=10,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert report.repaired_bytes == len(partial.encode("utf-8"))
    assert (report.already_done, report.written) == (1, 1)
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert [r["scenario"]["id"] for r in load_records(out)] == ["s00001", "s00003"]


def test_intact_file_is_not_touched(tmp_path):
    out = tmp_path / "raw.jsonl"
    out.write_text("", encoding="utf-8")
    report = synthesize(
        [],
        query_llm=ScriptedLLM([]),
        teacher=ScriptedLLM([]),
        out_path=out,
        batch_size=10,
        spent_usd=lambda: 0.0,
        budget_usd=1.0,
    )
    assert report.repaired_bytes == 0
