"""Decide whether one teacher trajectory is good enough to train on.

`validate_record` returns the reasons a record is rejected; an empty list
keeps it. Every reason is short and stable so `data_report.json` can count
them. Checks, in the spirit of astro-vlm-lora's "one number that does not fit":

- the run ended normally and the teacher never erred (no parse/schema/llm
  error, no fallback) -- we do not train on recoveries;
- the calls match the scenario's category and tools, and a tool only fails
  in a tool_error scenario (a mock rejecting the call means bad arguments);
- the user message uses the scenario's date phrase verbatim, and that
  relative date resolves to exactly what `dates.py` computes;
- no simplified characters or mainland wording (OpenCC s2twp round trip);
- the generated user message does not leak tool names.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from opencc import OpenCC

from twagent.dates import resolve_date_expr
from twagent.parsing import parse_output
from twagent.tools.schemas import tool_names

from .scenarios import Scenario

_BAD_EVENTS = {"parse_error", "schema_error", "llm_error", "fallback"}
_S2TWP = OpenCC("s2twp")

# Correct Taiwanese words that OpenCC's s2twp nonetheless rewrites (the pilot
# rejected these): 查看→檢視, 濃郁→濃鬱.
TW_OK_WORDS = ("查看", "濃郁")


def wording_problem(text: str) -> str | None:
    """A snippet showing simplified/mainland wording, or None if clean.

    台 vs 臺 is not a problem (both are Taiwanese usage), so it is normalised
    before the comparison. Allowlisted words (`TW_OK_WORDS`) are masked with
    "□" placeholders first, so OpenCC's s2twp cannot flag them as mainland
    wording while the rest of the snippet logic still runs unchanged.
    """
    normalised = text.replace("台", "臺")
    masked = normalised
    for word in TW_OK_WORDS:
        masked = masked.replace(word, "□" * len(word))
    converted = _S2TWP.convert(masked)
    if converted == masked:
        return None
    for i, (a, b) in enumerate(zip(masked, converted, strict=False)):
        if a != b:
            return f"{normalised[max(0, i - 4) : i + 4]} -> {converted[max(0, i - 4) : i + 4]}"
    return f"{normalised[-8:]} -> {converted[-8:]}"


def _calls_per_turn(result: dict[str, Any]) -> list[list[str]]:
    turns = []
    for event in result["trace"]:
        if event["type"] == "llm_output":
            names = [c.name for c in parse_output(event["raw"]).tool_calls]
            if names:
                turns.append(names)
    return turns


def _category_problems(s: Scenario, result: dict[str, Any]) -> list[str]:
    names = [c["name"] for c in result["tool_calls"]]
    turns = _calls_per_turn(result)
    final = result["final_text"]
    if s.category == "single" and names != list(s.tools):
        return [f"single: expected {list(s.tools)}, got {names}"]
    if s.category == "parallel" and (len(turns) != 1 or Counter(turns[0]) != Counter(s.tools)):
        return [f"parallel: expected one turn with {list(s.tools)}, got {turns}"]
    if s.category == "sequential" and (names != list(s.tools) or len(turns) < 2):
        return [f"sequential: expected {list(s.tools)} over >=2 turns, got {turns}"]
    if s.category in ("clarify", "no_tool") and names:
        return [f"{s.category}: expected no calls, got {names}"]
    if s.category == "clarify" and not ("？" in final or "?" in final):
        return ["clarify: reply is not a question"]
    if s.category == "tool_error":
        if not names or names[0] != s.tools[0]:
            return [f"tool_error: expected {s.tools[0]} first, got {names}"]
        if not any(e["type"] == "tool_error" for e in result["trace"]):
            return ["tool_error: tool never failed"]
    return []


def _date_problem(s: Scenario, result: dict[str, Any]) -> str | None:
    if not s.date_expr:
        return None
    expected = resolve_date_expr(s.date_expr, s.today).isoformat()
    for call in result["tool_calls"]:
        args = call["arguments"]
        got = args.get("date") or (args.get("start") or "")[:10] or None
        if got is None:
            continue
        if got != expected:
            return f"date: {s.date_expr} should be {expected}, got {got}"
    return None


def _unexpected_tool_error(s: Scenario, result: dict[str, Any]) -> str | None:
    if s.category == "tool_error":
        return None
    for event in result["trace"]:
        if event["type"] == "tool_error":
            what = event.get("error") or event.get("call", {}).get("name", "")
            return f"unexpected tool_error: {what}"
    return None


def validate_record(record: dict[str, Any]) -> list[str]:
    s = Scenario.from_dict(record["scenario"])
    result = record["result"]
    if result["stopped_reason"] != "answer":
        return [f"stopped: {result['stopped_reason']}"]
    problems: list[str] = []
    bad = sorted({e["type"] for e in result["trace"]} & _BAD_EVENTS)
    if bad:
        problems.append(f"teacher error: {', '.join(bad)}")
    if not result["final_text"].strip():
        problems.append("empty final reply")
    problems += _category_problems(s, result)
    if (p := _unexpected_tool_error(s, result)) is not None:
        problems.append(p)
    message = record["message"]
    if s.date_expr and s.date_expr not in message:
        problems.append(f"date phrase missing: {s.date_expr}")
    if (p := _date_problem(s, result)) is not None:
        problems.append(p)
    if any(name in message for name in tool_names()):
        problems.append("tool name leaked into user message")
    texts = [message] + [m["content"] for m in record["messages"] if m["role"] == "assistant"]
    for text in texts:
        if (w := wording_problem(text)) is not None:
            problems.append(f"wording: {w}")
            break
    return problems


def reason_key(problem: str) -> str:
    """Collapse a problem string to its counting bucket (text before ':')."""
    return problem.split(":", 1)[0]
