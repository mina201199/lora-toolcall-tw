import json
from datetime import date

from twagent.prompting import build_system_prompt, render_tools_block
from twagent.tools.schemas import TOOLS


def test_today_is_injected():
    prompt = build_system_prompt(date(2026, 9, 23))
    assert "今天是 2026-09-23（星期三）" in prompt


def test_every_tool_appears_once_as_unescaped_json():
    block = render_tools_block(TOOLS)
    for t in TOOLS:
        line = json.dumps(t, ensure_ascii=False)
        assert block.count(line) == 1
    assert "\\u" not in block  # ensure_ascii=False, like the template's tojson


def test_block_shape_matches_qwen_template():
    block = render_tools_block(TOOLS[:1])
    assert block.startswith("# Tools\n\nYou may call one or more functions")
    assert "<tools>\n{" in block
    assert block.endswith(
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def test_persona_and_tools_are_separated_by_a_blank_line():
    prompt = build_system_prompt(date(2026, 9, 23))
    persona, _, rest = prompt.partition("\n\n# Tools")
    assert persona.startswith("你是臺灣生活助理")
    assert rest
