"""Date arithmetic the model is expected to do, done deterministically.

Used to build the system prompt (today's date is injected, never read from
the clock inside the agent) and, later, to check synthetic labels: a sample
that says "下週五" must carry the date this module computes, or it is dropped.

Weeks run Monday to Sunday.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAY_ZH = "一二三四五六日"
_WEEKDAY_INDEX = {ch: i for i, ch in enumerate(WEEKDAY_ZH)} | {"天": 6}
_SMALL_NUMBERS = {
    "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}  # fmt: skip
_FIXED_OFFSETS = {"昨天": -1, "今天": 0, "明天": 1, "後天": 2, "大後天": 3}

_THIS_WEEK = re.compile(r"^(?:本|這)(?:週|周|禮拜|星期)([一二三四五六日天])$")
_NEXT_WEEK = re.compile(r"^下(?:週|周|禮拜|星期|個禮拜|個星期)([一二三四五六日天])$")
_BARE_WEEKDAY = re.compile(r"^(?:週|周|禮拜|星期)([一二三四五六日天])$")
_DAYS_LATER = re.compile(r"^(\d+|[一二兩三四五六七八九十])天(?:後|之後)$")


def format_today(d: date) -> str:
    """`2026-09-23（星期三）` -- the form injected into the system prompt."""
    return f"{d.isoformat()}（星期{WEEKDAY_ZH[d.weekday()]}）"


def resolve_date_expr(expr: str, today: date) -> date:
    """Resolve a relative Chinese date expression against `today`.

    Raises ValueError for anything outside the supported vocabulary, so a
    generator that invents a new phrasing is caught instead of guessed at.
    """
    s = expr.strip()
    if s in _FIXED_OFFSETS:
        return today + timedelta(days=_FIXED_OFFSETS[s])
    if m := _DAYS_LATER.match(s):
        n = m.group(1)
        return today + timedelta(days=int(n) if n.isdigit() else _SMALL_NUMBERS[n])
    monday = today - timedelta(days=today.weekday())
    if m := _THIS_WEEK.match(s):
        return monday + timedelta(days=_WEEKDAY_INDEX[m.group(1)])
    if m := _NEXT_WEEK.match(s):
        return monday + timedelta(days=7 + _WEEKDAY_INDEX[m.group(1)])
    if m := _BARE_WEEKDAY.match(s):
        # The next such day on or after today.
        delta = (_WEEKDAY_INDEX[m.group(1)] - today.weekday()) % 7
        return today + timedelta(days=delta)
    raise ValueError(f"unsupported date expression: {expr!r}")


def roc_year(d: date) -> int:
    return d.year - 1911


def invoice_period(d: date) -> str:
    """Taiwan uniform-invoice period containing `d`, e.g. `115-09-10`.

    Periods are two months long and always start on an odd month.
    """
    start = d.month if d.month % 2 == 1 else d.month - 1
    return f"{roc_year(d)}-{start:02d}-{start + 1:02d}"


def previous_invoice_period(d: date) -> str:
    """The period before the one containing `d` ("上一期")."""
    start = d.month if d.month % 2 == 1 else d.month - 1
    if start == 1:
        return f"{roc_year(d) - 1}-11-12"
    return f"{roc_year(d)}-{start - 2:02d}-{start - 1:02d}"
