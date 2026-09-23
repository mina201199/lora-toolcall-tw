import json
from pathlib import Path

import pytest

from twagent.cases import CATEGORIES, case_from_dict, load_cases, validate_case

ROOT = Path(__file__).resolve().parents[1]

GOOD = {
    "id": "t-1",
    "category": "single",
    "today": "2026-09-23",
    "messages": [{"role": "user", "content": "明天台中會下雨嗎"}],
    "expected": {
        "decision": "call",
        "calls": [{"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-09-24"}}],
    },
}


def with_(**overrides):
    d = json.loads(json.dumps(GOOD))
    d.update(overrides)
    return d


def test_good_case_is_valid():
    assert validate_case(case_from_dict(GOOD)) == []


def test_handwritten_seed_file_is_valid_and_covers_every_category():
    cases, errors = load_cases(ROOT / "data" / "handwritten_test.jsonl")
    assert errors == []
    assert {c.category for c in cases} == set(CATEGORIES)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"category": "weird"}, "unknown category"),
        ({"expected": {"decision": "answer", "calls": []}}, "requires decision"),
        ({"category": "parallel"}, "at least 2"),
        (
            {
                "category": "clarify",
                "expected": {"decision": "clarify", "calls": GOOD["expected"]["calls"]},
            },
            "must not expect",
        ),
        ({"messages": [{"role": "assistant", "content": "hi"}]}, "end with a user turn"),
        (
            {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]},
            "user/assistant",
        ),
        (
            {
                "expected": {
                    "decision": "call",
                    "calls": [{"name": "get_weather", "arguments": {"city": "台中"}}],
                }
            },
            "get_weather",
        ),
        ({"fail_tools": {"get_weather": "逾時"}}, "only allowed in tool_error"),
        ({"category": "tool_error"}, "must set fail_tools"),
        ({"category": "tool_error", "fail_tools": {"search_thsr": "逾時"}}, "never called"),
    ],
)
def test_invalid_cases(override, fragment):
    errors = validate_case(case_from_dict(with_(**override)))
    assert any(fragment in e for e in errors), errors


def test_load_reports_line_numbers_and_duplicates(tmp_path):
    p = tmp_path / "cases.jsonl"
    lines = [
        json.dumps(GOOD, ensure_ascii=False),
        "{not json",
        json.dumps(GOOD, ensure_ascii=False),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cases, errors = load_cases(p)
    assert len(cases) == 1
    assert any(e.startswith("line 2:") for e in errors)
    assert any("line 3" in e and "duplicate id" in e for e in errors)
