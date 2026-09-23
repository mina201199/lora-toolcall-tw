from datetime import date

import pytest

from twagent.dates import (
    format_today,
    invoice_period,
    previous_invoice_period,
    resolve_date_expr,
    roc_year,
)

WED = date(2026, 9, 23)  # a Wednesday
SUN = date(2026, 9, 27)


def test_format_today():
    assert format_today(WED) == "2026-09-23（星期三）"
    assert format_today(SUN) == "2026-09-27（星期日）"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("今天", date(2026, 9, 23)),
        ("明天", date(2026, 9, 24)),
        ("後天", date(2026, 9, 25)),
        ("大後天", date(2026, 9, 26)),
        ("昨天", date(2026, 9, 22)),
        ("3天後", date(2026, 9, 26)),
        ("十天後", date(2026, 10, 3)),
        ("兩天之後", date(2026, 9, 25)),
        ("這週五", date(2026, 9, 25)),
        ("本週一", date(2026, 9, 21)),  # already past, still this week
        ("這禮拜天", date(2026, 9, 27)),
        ("下週五", date(2026, 10, 2)),
        ("下禮拜一", date(2026, 9, 28)),
        ("下個星期日", date(2026, 10, 4)),
        ("週五", date(2026, 9, 25)),
        ("星期三", date(2026, 9, 23)),  # today counts
        ("禮拜二", date(2026, 9, 29)),  # already past -> next week
    ],
)
def test_resolve_date_expr(expr, expected):
    assert resolve_date_expr(expr, WED) == expected


def test_resolve_crosses_year_boundary():
    assert resolve_date_expr("明天", date(2026, 12, 31)) == date(2027, 1, 1)
    assert resolve_date_expr("下週一", date(2026, 12, 30)) == date(2027, 1, 4)


def test_next_week_from_sunday_is_the_following_week():
    # Sunday belongs to the Mon-Sun week that started six days earlier.
    assert resolve_date_expr("下週一", SUN) == date(2026, 9, 28)


def test_unsupported_expression_raises():
    with pytest.raises(ValueError, match="unsupported"):
        resolve_date_expr("過陣子", WED)


def test_roc_year():
    assert roc_year(WED) == 115


@pytest.mark.parametrize(
    ("d", "period", "previous"),
    [
        (date(2026, 9, 23), "115-09-10", "115-07-08"),
        (date(2026, 10, 1), "115-09-10", "115-07-08"),
        (date(2026, 1, 15), "115-01-02", "114-11-12"),
        (date(2026, 12, 31), "115-11-12", "115-09-10"),
    ],
)
def test_invoice_periods(d, period, previous):
    assert invoice_period(d) == period
    assert previous_invoice_period(d) == previous
