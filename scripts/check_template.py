"""Verify that our hand-rendered prompt equals the model's own chat template.

Needs a running llama-server (started with --jinja). For each scenario it asks
the server to render two requests via POST /apply-template:

  A. the "official" way: `tools` field + structured assistant `tool_calls`
  B. our way: tools rendered into the system text, tool calls as plain text

and diffs the two prompts. Any difference means training and serving would
see different text, so fix `prompting.py` / `parsing.render_tool_calls`
before going further.

    python scripts/check_template.py --server http://localhost:8080
    python scripts/check_template.py --save tests/fixtures   # record for CI
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import date
from pathlib import Path

import httpx

from twagent.parsing import ToolCall, render_tool_calls
from twagent.prompting import build_persona, build_system_prompt
from twagent.tools.schemas import TOOLS

TODAY = date(2026, 9, 23)
CALL = ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"})
CALL_2 = ToolCall("get_weather", {"city": "高雄市", "date": "2026-09-24"})
RESULT = json.dumps({"city": "臺中市", "condition": "多雲"}, ensure_ascii=False)
RESULT_2 = json.dumps({"city": "高雄市", "condition": "晴"}, ensure_ascii=False)


def _official_call(calls: list[ToolCall], content: str = "") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                },
            }
            for i, c in enumerate(calls)
        ],
    }


def _ours_call(calls: list[ToolCall], content: str = "") -> dict:
    rendered = render_tool_calls(calls)
    return {"role": "assistant", "content": f"{content}\n{rendered}" if content else rendered}


def scenarios() -> dict[str, tuple[dict, dict]]:
    user = {"role": "user", "content": "明天台中會下雨嗎"}
    official_system = {"role": "system", "content": build_persona(TODAY)}
    ours_system = {"role": "system", "content": build_system_prompt(TODAY)}

    def tool(content: str, i: int, official: bool) -> dict:
        msg = {"role": "tool", "content": content}
        if official:
            msg["tool_call_id"] = f"call_{i}"
        return msg

    both = [CALL, CALL_2]
    answer = {"role": "assistant", "content": "明天臺中多雲。"}
    follow_up = {"role": "user", "content": "那後天呢？"}
    return {
        "first_turn": (
            {"messages": [official_system, user], "tools": TOOLS},
            {"messages": [ours_system, user]},
        ),
        "after_tool_call": (
            {
                "messages": [official_system, user, _official_call([CALL]), tool(RESULT, 0, True)],
                "tools": TOOLS,
            },
            {"messages": [ours_system, user, _ours_call([CALL]), tool(RESULT, 0, False)]},
        ),
        "parallel_calls": (
            {
                "messages": [
                    official_system,
                    user,
                    _official_call(both),
                    tool(RESULT, 0, True),
                    tool(RESULT_2, 1, True),
                ],
                "tools": TOOLS,
            },
            {
                "messages": [
                    ours_system,
                    user,
                    _ours_call(both),
                    tool(RESULT, 0, False),
                    tool(RESULT_2, 1, False),
                ]
            },
        ),
        "text_call_answer_followup": (
            {
                "messages": [
                    official_system,
                    user,
                    _official_call([CALL], "我幫你查一下。"),
                    tool(RESULT, 0, True),
                    answer,
                    follow_up,
                ],
                "tools": TOOLS,
            },
            {
                "messages": [
                    ours_system,
                    user,
                    _ours_call([CALL], "我幫你查一下。"),
                    tool(RESULT, 0, False),
                    answer,
                    follow_up,
                ]
            },
        ),
    }


def render(http: httpx.Client, server: str, body: dict) -> str:
    resp = http.post(f"{server.rstrip('/')}/apply-template", json=body)
    resp.raise_for_status()
    return resp.json()["prompt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument(
        "--save", type=Path, help="write each official rendering to DIR/qwen_<name>.txt"
    )
    args = ap.parse_args()
    failed = False
    with httpx.Client(timeout=30) as http:
        for name, (official, ours) in scenarios().items():
            a, b = render(http, args.server, official), render(http, args.server, ours)
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                (args.save / f"qwen_{name}.txt").write_text(a, encoding="utf-8", newline="")
            if a == b:
                print(f"[same] {name}")
                continue
            failed = True
            print(f"[DIFF] {name}")
            sys.stdout.writelines(
                difflib.unified_diff(
                    a.splitlines(keepends=True), b.splitlines(keepends=True), "official", "ours"
                )
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
