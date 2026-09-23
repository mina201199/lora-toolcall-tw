"""The renderer must reproduce the recorded official Qwen template output.

The four fixtures were produced by scripts/check_template.py against the
model's own chat template (GGUF-embedded, via llama.cpp /apply-template); each
test below rebuilds the same conversation in our message format.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from twagent.parsing import ToolCall, render_tool_calls
from twagent.prompting import build_system_prompt
from twagent.render import Segment, render_chat, to_text

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SYSTEM = {"role": "system", "content": build_system_prompt(date(2026, 9, 23))}
USER = {"role": "user", "content": "明天台中會下雨嗎"}
CALL = ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"})
CALL_2 = ToolCall("get_weather", {"city": "高雄市", "date": "2026-09-24"})
RESULT = json.dumps({"city": "臺中市", "condition": "多雲"}, ensure_ascii=False)
RESULT_2 = json.dumps({"city": "高雄市", "condition": "晴"}, ensure_ascii=False)


def tool(content):
    return {"role": "tool", "content": content}


def assistant(content):
    return {"role": "assistant", "content": content}


CONVERSATIONS = {
    "first_turn": [SYSTEM, USER],
    "after_tool_call": [SYSTEM, USER, assistant(render_tool_calls([CALL])), tool(RESULT)],
    "parallel_calls": [
        SYSTEM,
        USER,
        assistant(render_tool_calls([CALL, CALL_2])),
        tool(RESULT),
        tool(RESULT_2),
    ],
    "text_call_answer_followup": [
        SYSTEM,
        USER,
        assistant("我幫你查一下。\n" + render_tool_calls([CALL])),
        tool(RESULT),
        assistant("明天臺中多雲。"),
        {"role": "user", "content": "那後天呢？"},
    ],
}


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
def test_matches_official_template(name):
    expected = (FIXTURES / f"qwen_{name}.txt").read_text(encoding="utf-8")
    rendered = to_text(render_chat(CONVERSATIONS[name], add_generation_prompt=True))
    assert rendered == expected


def test_only_assistant_content_is_trained():
    segments = render_chat(CONVERSATIONS["text_call_answer_followup"])
    trained = [s.text for s in segments if s.train]
    assert trained == [
        "我幫你查一下。\n" + render_tool_calls([CALL]) + "<|im_end|>",
        "明天臺中多雲。<|im_end|>",
    ]
    context = "".join(s.text for s in segments if not s.train)
    assert "<tool_response>" in context and "明天台中會下雨嗎" in context


def test_generation_prompt_is_not_trained():
    last = render_chat([SYSTEM, USER], add_generation_prompt=True)[-1]
    assert last == Segment("<|im_start|>assistant\n", train=False)


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="unknown role"):
        render_chat([{"role": "developer", "content": "x"}])
