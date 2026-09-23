"""Test cases: the format shared by the handwritten and synthetic test sets.

One JSON object per line:

    {"id": "hw-001", "category": "single", "today": "2026-09-23",
     "messages": [{"role": "user", "content": "明天台中會下雨嗎"}],
     "expected": {"decision": "call",
                  "calls": [{"name": "get_weather",
                             "arguments": {"city": "臺中市", "date": "2026-09-24"}}]},
     "fail_tools": {}, "note": ""}

`expected.calls` lists every valid call the agent should make, in order,
across all steps. `fail_tools` makes MockTools raise for those tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from twagent.llm import Message
from twagent.parsing import ToolCall
from twagent.tools.schemas import validate_arguments

CATEGORIES = ("single", "parallel", "sequential", "clarify", "no_tool", "tool_error")
EXPECTED_DECISION = {
    "single": "call",
    "parallel": "call",
    "sequential": "call",
    "clarify": "clarify",
    "no_tool": "answer",
    "tool_error": "call",
}
MIN_CALLS = {"single": 1, "parallel": 2, "sequential": 2, "tool_error": 1}


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    today: date
    messages: list[Message]
    decision: str
    calls: list[ToolCall]
    fail_tools: dict[str, str] = field(default_factory=dict)
    note: str = ""


def case_from_dict(d: dict[str, Any]) -> Case:
    """Build a Case; raises ValueError / KeyError / TypeError on bad structure."""
    expected = d["expected"]
    return Case(
        id=str(d["id"]),
        category=d["category"],
        today=date.fromisoformat(d["today"]),
        messages=[{"role": m["role"], "content": m["content"]} for m in d["messages"]],
        decision=expected["decision"],
        calls=[ToolCall(c["name"], c.get("arguments", {})) for c in expected.get("calls", [])],
        fail_tools=dict(d.get("fail_tools", {})),
        note=d.get("note", ""),
    )


def validate_case(c: Case) -> list[str]:
    errors: list[str] = []
    if c.category not in CATEGORIES:
        return [f"unknown category {c.category!r}"]
    if c.decision != EXPECTED_DECISION[c.category]:
        errors.append(f"category {c.category} requires decision {EXPECTED_DECISION[c.category]!r}")
    if not c.messages or c.messages[-1]["role"] != "user":
        errors.append("messages must be non-empty and end with a user turn")
    if any(m["role"] not in ("user", "assistant") for m in c.messages):
        errors.append("messages may only contain user/assistant turns")
    need = MIN_CALLS.get(c.category, 0)
    if need and len(c.calls) < need:
        errors.append(f"category {c.category} needs at least {need} call(s), got {len(c.calls)}")
    if not need and c.calls:
        errors.append(f"category {c.category} must not expect tool calls")
    for i, call in enumerate(c.calls, start=1):
        errors.extend(
            f"call #{i} {call.name}: {e}" for e in validate_arguments(call.name, call.arguments)
        )
    if c.category == "tool_error":
        if not c.fail_tools:
            errors.append("tool_error cases must set fail_tools")
        elif not set(c.fail_tools) <= {call.name for call in c.calls}:
            errors.append("fail_tools names a tool that is never called")
    elif c.fail_tools:
        errors.append("fail_tools is only allowed in tool_error cases")
    return errors


def load_cases(path: str | Path) -> tuple[list[Case], list[str]]:
    """Parse a JSONL file. Returns (valid cases, errors prefixed with line numbers)."""
    cases: list[Case] = []
    errors: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                case = case_from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                errors.append(f"line {lineno}: cannot parse ({type(e).__name__}: {e})")
                continue
            problems = validate_case(case)
            if case.id in seen:
                problems.append(f"duplicate id {case.id!r}")
            seen.add(case.id)
            if problems:
                errors.extend(f"line {lineno} ({case.id}): {p}" for p in problems)
            else:
                cases.append(case)
    return cases, errors
