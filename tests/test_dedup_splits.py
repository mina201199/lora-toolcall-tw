from collections import Counter

import pytest

from twagent.data.dedup import dedupe, matches_any
from twagent.data.scenarios import build_scenarios
from twagent.data.splits import DataLeakageError, SplitResult, group_of, split_by_group


def test_dedupe_ignores_punctuation_and_keeps_first():
    texts = ["明天台中會下雨嗎？", "明天台中會下雨嗎", "後天高雄天氣如何", "明天 台中 會下雨嗎!!"]
    assert dedupe(texts, threshold=0.8) == [0, 2]


def test_dedupe_keeps_distinct_messages():
    texts = ["幫我查明天台北到左營的高鐵", "我在台大附近哪裡有YouBike", "發票12345678有沒有中"]
    assert dedupe(texts, threshold=0.8) == [0, 1, 2]


def test_matches_any():
    assert matches_any(["明天台中會下雨嗎", "今天吃什麼"], ["明天台中會下雨嗎？"], 0.8) == [0]
    assert matches_any([], ["x"], 0.8) == [] and matches_any(["x"], [], 0.8) == []


def fake_records(n=600, seed=1):
    # Messages unique per scenario so only group structure matters.
    return [
        {"scenario": s.to_dict(), "message": f"{s.id} 的訊息內容 {i} 號，來自 {s.group}"}
        for i, s in enumerate(build_scenarios(n, seed=seed))
    ]


def test_split_is_group_disjoint_complete_and_stratified():
    records = fake_records()
    split = split_by_group(records, seed=7)
    parts = {"train": split.train, "valid": split.valid, "test": split.test}
    assert sum(len(p) for p in parts.values()) == len(records)
    groups = {k: {group_of(r) for r in v} for k, v in parts.items()}
    assert not (groups["train"] & groups["test"]) and not (groups["train"] & groups["valid"])
    for part in parts.values():
        assert set(Counter(r["scenario"]["category"] for r in part)) == {
            "single", "parallel", "sequential", "clarify", "no_tool", "tool_error"
        }  # fmt: skip
    assert 0.08 < len(split.test) / len(records) < 0.2


def test_split_is_deterministic():
    records = fake_records(300)
    a, b = split_by_group(records, seed=7), split_by_group(records, seed=7)
    assert [r["message"] for r in a.test] == [r["message"] for r in b.test]


def test_shared_group_cannot_be_constructed():
    records = fake_records(100)
    with pytest.raises(DataLeakageError, match="groups in both"):
        SplitResult(train=records[:60], valid=[], test=records[50:])


def test_near_duplicate_across_groups_cannot_be_constructed():
    records = fake_records(100)
    a = next(r for r in records if r["scenario"]["category"] == "single")
    b = next(r for r in records if group_of(r) != group_of(a))
    b = {**b, "message": a["message"] + "！"}
    with pytest.raises(DataLeakageError, match="near-duplicate"):
        SplitResult(train=[a], valid=[], test=[b])
