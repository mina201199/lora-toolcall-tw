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
RESULT = json.dumps({"city": "臺中市", "condition": "多雲"}, ensure_ascii=False)


def scenarios() -> dict[str, tuple[dict, dict]]:
    user = {"role": "user", "content": "明天台中會下雨嗎"}
    official_system = {"role": "system", "content": build_persona(TODAY)}
    ours_system = {"role": "system", "content": build_system_prompt(TODAY)}
    official_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": CALL.name,
                    "arguments": json.dumps(CALL.arguments, ensure_ascii=False),
                },
            }
        ],
    }
    ours_call = {"role": "assistant", "content": render_tool_calls([CALL])}
    tool_msg = {"role": "tool", "tool_call_id": "call_0", "content": RESULT}
    ours_tool = {"role": "tool", "content": RESULT}
    return {
        "first_turn": (
            {"messages": [official_system, user], "tools": TOOLS},
            {"messages": [ours_system, user]},
        ),
        "after_tool_call": (
            {"messages": [official_system, user, official_call, tool_msg], "tools": TOOLS},
            {"messages": [ours_system, user, ours_call, ours_tool]},
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
