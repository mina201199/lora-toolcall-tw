import pytest

from twagent.tools.schemas import CITIES, THSR_STATIONS, TOOLS, tool_names, validate_arguments


def test_seven_tools_with_unique_names():
    assert tool_names() == [
        "get_weather",
        "search_thsr",
        "search_tra",
        "find_youbike",
        "check_invoice",
        "calendar_add",
        "calendar_list",
    ]
    assert len({t["function"]["name"] for t in TOOLS}) == len(TOOLS)


def test_reference_lists():
    assert len(CITIES) == 22 and len(set(CITIES)) == 22
    assert len(THSR_STATIONS) == 12
    # Official spelling only: the model must learn to normalise 台 -> 臺.
    assert not any("台" in c for c in CITIES + THSR_STATIONS)


def test_every_tool_forbids_extra_arguments():
    for t in TOOLS:
        assert t["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_weather", {"city": "臺中市", "date": "2026-09-24"}),
        ("search_thsr", {"origin": "臺北", "destination": "左營", "date": "2026-09-24"}),
        (
            "search_thsr",
            {
                "origin": "臺北",
                "destination": "臺南",
                "date": "2026-09-24",
                "depart_after": "07:30",
            },
        ),
        (
            "search_tra",
            {"origin": "花蓮", "destination": "臺北", "date": "2026-09-24", "train_type": "自強"},
        ),
        ("find_youbike", {"city": "臺北市", "near": "臺大"}),
        ("check_invoice", {"number": "12345678", "period": "115-07-08"}),
        (
            "calendar_add",
            {"title": "看牙醫", "start": "2026-09-24T14:00", "end": "2026-09-24T15:00"},
        ),
        ("calendar_list", {"date": "2026-09-24"}),
    ],
)
def test_valid_calls(name, args):
    assert validate_arguments(name, args) == []


@pytest.mark.parametrize(
    ("name", "args", "fragment"),
    [
        ("get_weather", {"city": "台中", "date": "2026-09-24"}, "city"),
        ("get_weather", {"city": "臺中市"}, "'date' is a required property"),
        ("get_weather", {"city": "臺中市", "date": "明天"}, "date"),
        ("get_weather", {"city": "臺中市", "date": "2026-09-24", "unit": "C"}, "unit"),
        ("search_thsr", {"origin": "高雄", "destination": "臺北", "date": "2026-09-24"}, "origin"),
        (
            "search_thsr",
            {
                "origin": "臺北",
                "destination": "左營",
                "date": "2026-09-24",
                "depart_after": "25:00",
            },
            "depart_after",
        ),
        ("check_invoice", {"number": "1234567", "period": "115-07-08"}, "number"),
        ("check_invoice", {"number": "12345678", "period": "115-08-09"}, "period"),
        (
            "calendar_add",
            {"title": "x", "start": "2026-09-24 14:00", "end": "2026-09-24T15:00"},
            "start",
        ),
        (
            "get_weather",
            {"city": "臺中市", "date": "2026-02-30"},
            "date: 2026-02-30 is not a real date/time",
        ),
        (
            "calendar_add",
            {"title": "x", "start": "2026-02-30T10:00", "end": "2026-09-24T15:00"},
            "start: 2026-02-30T10:00 is not a real date/time",
        ),
        (
            "calendar_add",
            {"title": "x", "start": "2026-09-24T15:00", "end": "2026-09-24T15:00"},
            "end: must be later than start",
        ),
        (
            "calendar_add",
            {"title": "x", "start": "2026-09-24T15:00", "end": "2026-09-24T14:00"},
            "end: must be later than start",
        ),
    ],
)
def test_invalid_calls(name, args, fragment):
    errors = validate_arguments(name, args)
    assert errors, "expected at least one error"
    assert any(fragment in e for e in errors), errors


def test_unknown_tool():
    errors = validate_arguments("book_flight", {})
    assert len(errors) == 1 and "unknown tool" in errors[0]


def test_non_object_arguments():
    assert "JSON object" in validate_arguments("get_weather", ["臺中市"])[0]
