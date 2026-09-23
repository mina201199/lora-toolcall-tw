"""The single definition of every tool the agent can call.

Synthesis, validation, evaluation, the system prompt and the server all read
from `TOOLS`. There is no second copy to drift.

Enum values use the official 臺 spelling; teaching the model to normalise
"台北" -> "臺北市" is part of the task.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from jsonschema import Draft202012Validator

CITIES = [
    "臺北市", "新北市", "桃園市", "臺中市", "臺南市", "高雄市",
    "基隆市", "新竹市", "嘉義市", "新竹縣", "苗栗縣", "彰化縣",
    "南投縣", "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣",
    "臺東縣", "澎湖縣", "金門縣", "連江縣",
]  # fmt: skip

THSR_STATIONS = [
    "南港", "臺北", "板橋", "桃園", "新竹", "苗栗",
    "臺中", "彰化", "雲林", "嘉義", "臺南", "左營",
]  # fmt: skip

TRA_TRAIN_TYPES = ["自強", "莒光", "區間快", "區間"]

DATE = {"type": "string", "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", "description": "YYYY-MM-DD"}
TIME = {
    "type": "string",
    "pattern": r"^([01][0-9]|2[0-3]):[0-5][0-9]$",
    "description": "HH:MM（24 小時制）",
}
DATETIME = {
    "type": "string",
    "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T([01][0-9]|2[0-3]):[0-5][0-9]$",
    "description": "YYYY-MM-DDTHH:MM（臺灣時間）",
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "get_weather",
        "查詢臺灣某縣市某一天的天氣預報。",
        {"city": {"type": "string", "enum": CITIES}, "date": DATE},
        ["city", "date"],
    ),
    _tool(
        "search_thsr",
        "查詢高鐵班次。",
        {
            "origin": {"type": "string", "enum": THSR_STATIONS},
            "destination": {"type": "string", "enum": THSR_STATIONS},
            "date": DATE,
            "depart_after": TIME,
        },
        ["origin", "destination", "date"],
    ),
    _tool(
        "search_tra",
        "查詢臺鐵（火車）班次。",
        {
            "origin": {"type": "string", "minLength": 1, "description": "臺鐵車站名稱"},
            "destination": {"type": "string", "minLength": 1, "description": "臺鐵車站名稱"},
            "date": DATE,
            "depart_after": TIME,
            "train_type": {"type": "string", "enum": TRA_TRAIN_TYPES},
        },
        ["origin", "destination", "date"],
    ),
    _tool(
        "find_youbike",
        "查詢某地點附近的 YouBike 站點與可借車輛數。",
        {
            "city": {"type": "string", "enum": CITIES},
            "near": {"type": "string", "minLength": 1, "description": "地標、路名或區域"},
        },
        ["city", "near"],
    ),
    _tool(
        "check_invoice",
        "統一發票對獎。",
        {
            "number": {
                "type": "string",
                "pattern": r"^[0-9]{8}$",
                "maxLength": 8,
                "description": "8 碼發票號碼",
            },
            "period": {
                "type": "string",
                "pattern": r"^[0-9]{3}-(01-02|03-04|05-06|07-08|09-10|11-12)$",
                "maxLength": 9,
                "description": "民國年-起月-迄月，例如 115-07-08",
            },
        },
        ["number", "period"],
    ),
    _tool(
        "calendar_add",
        "在使用者的行事曆新增一筆行程。",
        {
            "title": {"type": "string", "minLength": 1},
            "start": DATETIME,
            "end": DATETIME,
            "location": {"type": "string"},
        },
        ["title", "start", "end"],
    ),
    _tool(
        "calendar_list",
        "列出使用者行事曆某一天的行程。",
        {"date": DATE},
        ["date"],
    ),
]

_BY_NAME = {t["function"]["name"]: t for t in TOOLS}
_VALIDATORS = {
    name: Draft202012Validator(t["function"]["parameters"]) for name, t in _BY_NAME.items()
}


def tool_names() -> list[str]:
    return list(_BY_NAME)


def get_tool(name: str) -> dict[str, Any]:
    return _BY_NAME[name]


def validate_arguments(name: str, arguments: Any) -> list[str]:
    """Return human-readable problems with a call; empty list means valid.

    The messages go back to the model verbatim as a tool result, so they are
    meant to be read by it.
    """
    if name not in _VALIDATORS:
        return [f"unknown tool: {name!r}; available: {', '.join(_BY_NAME)}"]
    if not isinstance(arguments, dict):
        return [f"arguments must be a JSON object, got {type(arguments).__name__}"]
    errors = sorted(_VALIDATORS[name].iter_errors(arguments), key=lambda e: list(e.path))
    out = []
    for e in errors:
        where = ".".join(str(p) for p in e.path) or "(root)"
        out.append(f"{where}: {e.message}")
    if out:
        return out
    return _semantic_errors(name, arguments)


def _semantic_errors(name: str, arguments: dict[str, Any]) -> list[str]:
    """Checks the JSON Schema pass cannot express: real calendar dates, and
    `calendar_add.end` being later than `start`. Only run once the arguments
    already match their JSON Schema (right types/patterns), so `.get`/casts
    below are safe.
    """
    properties = _BY_NAME[name]["function"]["parameters"]["properties"]
    out = []
    for prop, value in arguments.items():
        schema = properties.get(prop)
        if schema is DATE:
            try:
                date.fromisoformat(value)
            except ValueError:
                out.append(f"{prop}: {value} is not a real date/time")
        elif schema is DATETIME:
            try:
                datetime.fromisoformat(value)
            except ValueError:
                out.append(f"{prop}: {value} is not a real date/time")
        elif schema is TIME:
            # The pattern's `$` also matches before a trailing newline.
            try:
                datetime.strptime(value, "%H:%M")
            except ValueError:
                out.append(f"{prop}: {value!r} is not a valid HH:MM time")
    if out:
        return out
    if name == "calendar_add":
        start, end = arguments.get("start"), arguments.get("end")
        if (
            start is not None
            and end is not None
            and datetime.fromisoformat(end) <= datetime.fromisoformat(start)
        ):
            out.append("end: must be later than start")
    return out
