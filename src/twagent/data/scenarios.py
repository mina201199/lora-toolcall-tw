"""The scenario matrix every synthetic sample is drawn from (spec §5.1-5.2).

A scenario fixes what a sample must be -- category, tools, speaking style,
the relative-date phrase to use and the "today" it is relative to -- before
any model writes a word. Validation later checks the teacher's trajectory
against the scenario, so a sample that drifted from its brief is dropped.

`group` is the unit of train/valid/test splitting: every sample of one
(category, tools, style) cell lands in the same split, so the test set only
contains combinations the model never trained on.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from itertools import islice
from typing import Any

from twagent.dates import resolve_date_expr

CATEGORY_SHARE = {
    "single": 0.35,
    "parallel": 0.15,
    "sequential": 0.15,
    "clarify": 0.15,
    "no_tool": 0.15,
    "tool_error": 0.05,
}

STYLES = (
    "口語隨興",
    "有錯字或注音文",
    "中英夾雜",
    "很長而且繞圈子",
    "非常簡短",
    "客氣正式",
)

_ALL_TOOLS = (
    "get_weather",
    "search_thsr",
    "search_tra",
    "find_youbike",
    "check_invoice",
    "calendar_add",
    "calendar_list",
)

COMBOS: dict[str, list[tuple[str, ...]]] = {
    "single": [(t,) for t in _ALL_TOOLS],
    "parallel": [
        ("get_weather", "get_weather"),
        ("search_thsr", "get_weather"),
        ("find_youbike", "get_weather"),
        ("check_invoice", "check_invoice"),
        ("calendar_list", "get_weather"),
        ("search_tra", "search_thsr"),
    ],
    # Later steps need the earlier step's result (spec §5.1).
    "sequential": [
        ("search_thsr", "calendar_add"),
        ("search_tra", "calendar_add"),
        ("calendar_list", "calendar_add"),
        ("get_weather", "calendar_add"),
    ],
    "clarify": [(t,) for t in _ALL_TOOLS],
    "no_tool": [()],
    "tool_error": [(t,) for t in _ALL_TOOLS if t != "calendar_list"],
}

NO_TOOL_KINDS = ("閒聊", "一般常識問題", "不支援的請求（訂票、付款、叫車等）")

DATED_TOOLS = {"get_weather", "search_thsr", "search_tra", "calendar_add", "calendar_list"}
DATE_EXPRS = ("今天", "明天", "後天", "大後天", "這週五", "下週一", "下禮拜三", "週六", "3天後")
FAILURES = ("連線逾時", "服務暫時無法使用", "上游資料格式錯誤")


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    tools: tuple[str, ...]
    style: str
    today: date
    date_expr: str | None = None
    kind: str = ""  # no_tool only
    failure: str = ""  # tool_error only: message the first tool fails with

    @property
    def group(self) -> str:
        what = "+".join(self.tools) if self.tools else self.kind
        return f"{self.category}|{what}|{self.style}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tools"] = list(self.tools)
        d["today"] = self.today.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        return cls(**{**d, "tools": tuple(d["tools"]), "today": date.fromisoformat(d["today"])})


# Integer weights per block of 20 scenarios: 7/3/3/3/3/1.
_BLOCK = 20
_WEIGHTS = {c: round(share * _BLOCK) for c, share in CATEGORY_SHARE.items()}
assert sum(_WEIGHTS.values()) == _BLOCK, _WEIGHTS


def _category_stream() -> Iterator[str]:
    """Smooth weighted round-robin over `_WEIGHTS`, forever.

    Categories are interleaved (7/3/3/3/3/1 spread out, not in runs), and every
    block of `_BLOCK` consecutive items has exactly the weighted counts.
    """
    current = dict.fromkeys(_WEIGHTS, 0)
    while True:
        for c, w in _WEIGHTS.items():
            current[c] += w
        pick = max(current, key=current.__getitem__)  # ties: CATEGORY_SHARE order
        current[pick] -= _BLOCK
        yield pick


def _cells(category: str) -> list[tuple[tuple[str, ...], str, str]]:
    if category == "no_tool":
        return [((), kind, style) for kind in NO_TOOL_KINDS for style in STYLES]
    return [(combo, "", style) for combo in COMBOS[category] for style in STYLES]


def _future_date_exprs(today: date) -> list[str]:
    """DATE_EXPRS that resolve to `today` or later (e.g. not 這週五 on a Saturday)."""
    return [e for e in DATE_EXPRS if resolve_date_expr(e, today) >= today]


def build_scenarios(total: int, seed: int) -> list[Scenario]:
    """The first `total` items of one deterministic scenario stream for `seed`.

    The stream does not depend on `total`, so a shorter run is an exact prefix
    of a longer one (same ids, same content) and an interrupted run's prefix
    still covers every category. Categories are interleaved by weighted
    round-robin, so any 20 consecutive scenarios have exact CATEGORY_SHARE
    proportions. Within a category, cells (combo x style, or kind x style) are
    visited round-robin in a shuffled order, so every cell gets used before
    any cell is used twice.
    """
    rng = random.Random(seed)
    year_start = date(2026, 1, 1)
    cells = {category: _cells(category) for category in CATEGORY_SHARE}
    for category in CATEGORY_SHARE:  # fixed order, so the shuffles are deterministic
        rng.shuffle(cells[category])
    used = dict.fromkeys(CATEGORY_SHARE, 0)
    out: list[Scenario] = []
    for category in islice(_category_stream(), total):
        own = cells[category]
        tools, kind, style = own[used[category] % len(own)]
        used[category] += 1
        # clarify asks the user to leave a required detail (often the date) out.
        dated = category != "clarify" and any(t in DATED_TOOLS for t in tools)
        today = year_start + timedelta(days=rng.randrange(365))
        out.append(
            Scenario(
                id=f"s{len(out):05d}",
                category=category,
                tools=tools,
                style=style,
                today=today,
                date_expr=rng.choice(_future_date_exprs(today)) if dated else None,
                kind=kind,
                failure=rng.choice(FAILURES) if category == "tool_error" else "",
            )
        )
    return out
