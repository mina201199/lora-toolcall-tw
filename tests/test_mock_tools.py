import json

import pytest

from twagent.tools.base import ToolError
from twagent.tools.mock import MockTools
from twagent.tools.schemas import tool_names

SAMPLE_CALLS = {
    "get_weather": {"city": "臺中市", "date": "2026-09-24"},
    "search_thsr": {
        "origin": "臺北",
        "destination": "左營",
        "date": "2026-09-24",
        "depart_after": "08:00",
    },
    "search_tra": {"origin": "花蓮", "destination": "臺北", "date": "2026-09-24"},
    "find_youbike": {"city": "臺北市", "near": "臺大"},
    "check_invoice": {"number": "12345678", "period": "115-07-08"},
    "calendar_add": {"title": "看牙醫", "start": "2026-09-24T14:00", "end": "2026-09-24T15:00"},
    "calendar_list": {"date": "2026-09-24"},
}


def test_every_tool_has_a_mock():
    assert set(SAMPLE_CALLS) == set(tool_names())
    tools = MockTools()
    for name, args in SAMPLE_CALLS.items():
        result = tools.call(name, args)
        json.dumps(result, ensure_ascii=False)  # must be serialisable


@pytest.mark.parametrize("name", [n for n in SAMPLE_CALLS if not n.startswith("calendar")])
def test_same_call_same_result_across_instances(name):
    assert MockTools().call(name, SAMPLE_CALLS[name]) == MockTools().call(name, SAMPLE_CALLS[name])


def test_different_arguments_give_different_weather():
    a = MockTools().call("get_weather", {"city": "臺中市", "date": "2026-09-24"})
    b = MockTools().call("get_weather", {"city": "臺北市", "date": "2026-09-24"})
    assert a != b


def test_thsr_respects_depart_after():
    result = MockTools().call("search_thsr", SAMPLE_CALLS["search_thsr"])
    assert len(result["trains"]) == 3
    assert all(t["depart"] >= "08:00" for t in result["trains"])


def test_same_station_is_a_tool_error():
    with pytest.raises(ToolError, match="起訖站相同"):
        MockTools().call(
            "search_thsr", {"origin": "臺北", "destination": "臺北", "date": "2026-09-24"}
        )


def test_calendar_state_is_inspectable():
    tools = MockTools()
    tools.call("calendar_add", SAMPLE_CALLS["calendar_add"])
    tools.call(
        "calendar_add",
        {
            "title": "開會",
            "start": "2026-09-25T10:00",
            "end": "2026-09-25T11:00",
            "location": "公司",
        },
    )
    assert [e["title"] for e in tools.events()] == ["看牙醫", "開會"]
    listed = tools.call("calendar_list", {"date": "2026-09-24"})
    assert [e["title"] for e in listed["events"]] == ["看牙醫"]


def test_calendar_rejects_end_before_start():
    with pytest.raises(ToolError, match="結束時間"):
        MockTools().call(
            "calendar_add", {"title": "x", "start": "2026-09-24T15:00", "end": "2026-09-24T14:00"}
        )


def test_injected_failure():
    tools = MockTools(failures={"search_thsr": "連線逾時"})
    with pytest.raises(ToolError, match="連線逾時"):
        tools.call("search_thsr", SAMPLE_CALLS["search_thsr"])
    # Other tools are unaffected.
    tools.call("get_weather", SAMPLE_CALLS["get_weather"])
