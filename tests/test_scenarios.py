from collections import Counter

import pytest

from twagent.data.scenarios import (
    CATEGORY_SHARE,
    COMBOS,
    DATE_EXPRS,
    STYLES,
    Scenario,
    build_scenarios,
)
from twagent.dates import resolve_date_expr
from twagent.tools.schemas import tool_names


def test_proportions():
    counts = Counter(s.category for s in build_scenarios(1000, seed=1))
    assert counts == {
        "single": 350,
        "parallel": 150,
        "sequential": 150,
        "clarify": 150,
        "no_tool": 150,
        "tool_error": 50,
    }


def test_any_total_is_honoured():
    scenarios = build_scenarios(7, seed=1)
    assert len(scenarios) == 7
    assert sum(CATEGORY_SHARE.values()) == pytest.approx(1.0)


def test_prefix_stable_across_totals():
    # An interrupted or budget-stopped run keeps a prefix; a different --total
    # must not reassign ids or reshuffle what that prefix contains.
    short, long = build_scenarios(500, seed=1), build_scenarios(1300, seed=1)
    assert long[:500] == short


def test_every_block_of_20_has_exact_proportions():
    scenarios = build_scenarios(200, seed=1)
    assert set(Counter(s.category for s in scenarios[:20])) == set(CATEGORY_SHARE)
    for i in range(0, 200, 20):
        counts = Counter(s.category for s in scenarios[i : i + 20])
        assert counts == {c: round(share * 20) for c, share in CATEGORY_SHARE.items()}


def test_ids_follow_stream_order():
    assert [s.id for s in build_scenarios(30, seed=1)] == [f"s{i:05d}" for i in range(30)]


def test_deterministic_and_seed_sensitive():
    a, b = build_scenarios(200, seed=3), build_scenarios(200, seed=3)
    assert a == b
    assert a != build_scenarios(200, seed=4)


def test_ids_unique():
    ids = [s.id for s in build_scenarios(500, seed=1)]
    assert len(set(ids)) == len(ids)


def test_every_cell_used_before_any_repeats():
    scenarios = [s for s in build_scenarios(1000, seed=1) if s.category == "single"]
    cells = Counter((s.tools, s.style) for s in scenarios)
    assert len(cells) == len(COMBOS["single"]) * len(STYLES)
    assert max(cells.values()) - min(cells.values()) <= 1


def test_combos_only_use_real_tools():
    names = set(tool_names())
    for combos in COMBOS.values():
        for combo in combos:
            assert set(combo) <= names


def test_date_expr_only_for_dated_tools_and_resolvable():
    dated_tools = {"get_weather", "search_thsr", "search_tra", "calendar_add", "calendar_list"}
    for s in build_scenarios(500, seed=2):
        dated = bool(dated_tools & set(s.tools)) and s.category != "clarify"
        assert (s.date_expr is not None) == dated, s
        if s.date_expr is not None:
            assert s.date_expr in DATE_EXPRS
            resolve_date_expr(s.date_expr, s.today)  # must not raise


def test_clarify_never_gets_a_date_expr():
    # The clarify brief asks the user to leave a required detail out -- often
    # the date -- so forcing a date phrase would contradict it.
    clarify = [s for s in build_scenarios(500, seed=2) if s.category == "clarify"]
    assert clarify and all(s.date_expr is None for s in clarify)


def test_category_specific_fields():
    for s in build_scenarios(500, seed=2):
        assert bool(s.kind) == (s.category == "no_tool")
        assert bool(s.failure) == (s.category == "tool_error")
        assert (s.tools == ()) == (s.category == "no_tool")


def test_today_within_2026():
    for s in build_scenarios(300, seed=5):
        assert s.today.year == 2026


def test_group_and_round_trip():
    s = build_scenarios(50, seed=1)[0]
    assert s.group.startswith(f"{s.category}|")
    assert Scenario.from_dict(s.to_dict()) == s


def test_group_ignores_today_and_date():
    a = Scenario(
        "a", "single", ("get_weather",), "口語隨興", build_scenarios(1, 1)[0].today, "明天"
    )
    b = Scenario("b", "single", ("get_weather",), "口語隨興", a.today.replace(month=1), "後天")
    assert a.group == b.group == "single|get_weather|口語隨興"


def test_date_expr_never_points_to_the_past():
    dated = [s for s in build_scenarios(600, seed=2) if s.date_expr is not None]
    assert len(dated) > 200
    for s in dated:
        assert resolve_date_expr(s.date_expr, s.today) >= s.today, s


def test_ambiguous_weekday_phrases_never_land_on_today():
    # e.g. 這週五 on a Friday or 週六 on a Saturday reads as ambiguous ("which
    # Saturday?", per the pilot's teacher) -- only the explicit 今天 may land
    # on today.
    dated = [s for s in build_scenarios(600, seed=2) if s.date_expr is not None]
    assert len(dated) > 200
    for s in dated:
        if s.date_expr == "今天":
            continue
        assert resolve_date_expr(s.date_expr, s.today) > s.today, s
