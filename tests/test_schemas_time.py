import pytest

from twagent.tools.schemas import validate_arguments

BASE = {"origin": "臺北", "destination": "左營", "date": "2026-09-24"}


@pytest.mark.parametrize("value", ["14:30\n", "24:00", "7:30"])
def test_invalid_depart_after(value):
    errors = validate_arguments("search_thsr", {**BASE, "depart_after": value})
    assert any(e.startswith("depart_after:") for e in errors), errors


@pytest.mark.parametrize("value", ["00:00", "07:30", "23:59"])
def test_valid_depart_after(value):
    assert validate_arguments("search_thsr", {**BASE, "depart_after": value}) == []


def test_tra_depart_after_trailing_newline():
    args = {
        "origin": "花蓮",
        "destination": "臺北",
        "date": "2026-09-24",
        "depart_after": "08:00\n",
    }
    assert any("depart_after" in e for e in validate_arguments("search_tra", args))
