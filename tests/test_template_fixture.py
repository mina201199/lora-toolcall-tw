"""Lock in what scripts/check_template.py verified against the real server.

The fixtures are the model's own chat-template rendering, recorded once with
`python scripts/check_template.py --save tests/fixtures`. If someone edits the
persona, a schema or the tool-call rendering, these tests say whether the
official template would still produce the same text -- without a GPU.
Re-record the fixtures whenever `TOOLS` or `PERSONA` change on purpose.
"""

import json
from datetime import date
from pathlib import Path

from twagent.parsing import ToolCall, render_tool_calls
from twagent.prompting import build_system_prompt

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TODAY = date(2026, 9, 23)
CALL = ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"})
RESULT = json.dumps({"city": "臺中市", "condition": "多雲"}, ensure_ascii=False)


def read(name: str) -> str:
    return (FIXTURES / f"qwen_{name}.txt").read_text(encoding="utf-8")


def test_system_prompt_matches_official_rendering():
    assert f"<|im_start|>system\n{build_system_prompt(TODAY)}<|im_end|>\n" in read("first_turn")


def test_tool_call_rendering_matches_official_rendering():
    assert render_tool_calls([CALL]) in read("after_tool_call")


def test_tool_result_wrapping():
    # Documents how the template wraps a `tool` message; the training
    # formatter (next plan) must produce the same thing.
    assert f"<tool_response>\n{RESULT}\n</tool_response>" in read("after_tool_call")
