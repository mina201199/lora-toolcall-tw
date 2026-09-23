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
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

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


def _counts(total: int) -> dict[str, int]:
    counts = {c: int(total * share) for c, share in CATEGORY_SHARE.items()}
    counts["single"] += total - sum(counts.values())  # rounding remainder
    return counts


def build_scenarios(total: int, seed: int) -> list[Scenario]:
    """`total` scenarios in CATEGORY_SHARE proportions, deterministic in `seed`.

    Within a category, cells (combo x style, or kind x style) are visited
    round-robin in a shuffled order, so every cell gets used before any cell
    is used twice.
    """
    rng = random.Random(seed)
    year_start = date(2026, 1, 1)
    out: list[Scenario] = []
    for category, n in _counts(total).items():
        if category == "no_tool":
            cells = [((), kind, style) for kind in NO_TOOL_KINDS for style in STYLES]
        else:
            cells = [(combo, "", style) for combo in COMBOS[category] for style in STYLES]
        rng.shuffle(cells)
        for i in range(n):
            tools, kind, style = cells[i % len(cells)]
            dated = any(t in DATED_TOOLS for t in tools)
            out.append(
                Scenario(
                    id=f"s{len(out):05d}",
                    category=category,
                    tools=tools,
                    style=style,
                    today=year_start + timedelta(days=rng.randrange(365)),
                    date_expr=rng.choice(DATE_EXPRS) if dated else None,
                    kind=kind,
                    failure=rng.choice(FAILURES) if category == "tool_error" else "",
                )
            )
    return out
