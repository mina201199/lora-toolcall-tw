"""Deterministic stand-ins for every tool.

Evaluation always runs against these, so a score can be reproduced exactly:
the same call returns the same result, seeded from a hash of the call itself.
The calendar is a real (in-memory) SQLite database so end-to-end tasks can be
scored by inspecting final state rather than by trusting the model's reply.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from datetime import datetime
from typing import Any

from twagent.tools.base import ToolError
from twagent.tools.schemas import THSR_STATIONS

_CONDITIONS = ["晴", "多雲", "陰", "短暫陣雨", "午後雷陣雨"]


def _rng(name: str, arguments: dict[str, Any]) -> random.Random:
    key = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True)
    return random.Random(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16))


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class MockTools:
    """`ToolExecutor` backed by seeded fake data and an in-memory calendar.

    `failures` maps a tool name to an error message; calling that tool raises
    `ToolError` with it. Used by the "tool error" test cases.
    """

    def __init__(self, failures: dict[str, str] | None = None) -> None:
        self.failures = dict(failures or {})
        self._db = sqlite3.connect(":memory:")
        self._db.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY, title TEXT, start TEXT, end TEXT, location TEXT)"
        )

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in self.failures:
            raise ToolError(self.failures[name])
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            raise ToolError(f"unknown tool: {name}")
        try:
            return handler(dict(arguments), _rng(name, arguments))
        except ValueError as e:
            raise ToolError(str(e)) from e

    def events(self) -> list[dict[str, Any]]:
        """Final calendar state, for end-to-end scoring."""
        rows = self._db.execute("SELECT id, title, start, end, location FROM events ORDER BY id")
        return [
            {"id": r[0], "title": r[1], "start": r[2], "end": r[3], "location": r[4]} for r in rows
        ]

    # -- handlers -----------------------------------------------------------

    def _get_weather(self, a: dict, rng: random.Random) -> dict:
        low = rng.randint(14, 27)
        return {
            "city": a["city"],
            "date": a["date"],
            "condition": rng.choice(_CONDITIONS),
            "temp_min": low,
            "temp_max": low + rng.randint(3, 8),
            "rain_probability": rng.choice([0, 10, 20, 30, 50, 70, 90]),
        }

    def _search_thsr(self, a: dict, rng: random.Random) -> dict:
        if a["origin"] == a["destination"]:
            raise ToolError("起訖站相同")
        hops = abs(THSR_STATIONS.index(a["origin"]) - THSR_STATIONS.index(a["destination"]))
        ride = 10 + hops * 9
        first = max(_minutes(a.get("depart_after", "06:00")), 6 * 60)
        first += rng.randint(0, 25)
        trains = []
        for i in range(3):
            dep = first + i * 30
            trains.append(
                {
                    "train_no": str(rng.randint(100, 1699)),
                    "depart": _hhmm(dep),
                    "arrive": _hhmm(dep + ride),
                }
            )
        return {
            "origin": a["origin"],
            "destination": a["destination"],
            "date": a["date"],
            "trains": trains,
        }

    def _search_tra(self, a: dict, rng: random.Random) -> dict:
        if a["origin"] == a["destination"]:
            raise ToolError("起訖站相同")
        ride = rng.randint(40, 240)
        first = max(_minutes(a.get("depart_after", "05:30")), 5 * 60 + 30) + rng.randint(0, 40)
        kind = a.get("train_type")
        trains = []
        for i in range(3):
            dep = first + i * 45
            trains.append(
                {
                    "train_no": str(rng.randint(100, 4999)),
                    "train_type": kind or rng.choice(["自強", "莒光", "區間"]),
                    "depart": _hhmm(dep),
                    "arrive": _hhmm(dep + ride),
                }
            )
        return {
            "origin": a["origin"],
            "destination": a["destination"],
            "date": a["date"],
            "trains": trains,
        }

    def _find_youbike(self, a: dict, rng: random.Random) -> dict:
        stations = [
            {
                "name": f"{a['near']}{suffix}",
                "available_bikes": rng.randint(0, 25),
                "empty_docks": rng.randint(0, 25),
                "distance_m": rng.randint(50, 600),
            }
            for suffix in ("站", "(東側)", "(捷運出口)")
        ]
        stations.sort(key=lambda s: s["distance_m"])
        return {"city": a["city"], "near": a["near"], "stations": stations}

    def _check_invoice(self, a: dict, rng: random.Random) -> dict:
        # The winning tail depends on the period only, like the real draw.
        winning_tail = f"{_rng('invoice-draw', {'period': a['period']}).randint(0, 999):03d}"
        won = a["number"].endswith(winning_tail)
        return {
            "period": a["period"],
            "number": a["number"],
            "result": "六獎（200 元）" if won else "未中獎",
        }

    def _calendar_add(self, a: dict, rng: random.Random) -> dict:
        start = datetime.fromisoformat(a["start"])
        end = datetime.fromisoformat(a["end"])
        if end <= start:
            raise ToolError("結束時間必須晚於開始時間")
        cur = self._db.execute(
            "INSERT INTO events (title, start, end, location) VALUES (?, ?, ?, ?)",
            (a["title"], a["start"], a["end"], a.get("location")),
        )
        return {"status": "created", "event_id": cur.lastrowid}

    def _calendar_list(self, a: dict, rng: random.Random) -> dict:
        rows = self._db.execute(
            "SELECT title, start, end, location FROM events WHERE substr(start, 1, 10) = ? ORDER BY start",
            (a["date"],),
        )
        events = [{"title": r[0], "start": r[1], "end": r[2], "location": r[3]} for r in rows]
        return {"date": a["date"], "events": events}
