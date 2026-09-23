from twagent.parsing import ToolCall, parse_output, render_tool_calls

WEATHER = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-09-24"}}\n</tool_call>'


def test_plain_answer():
    out = parse_output("你好！有什麼可以幫你？")
    assert out.ok and out.tool_calls == () and out.text == "你好！有什麼可以幫你？"


def test_single_call():
    out = parse_output(WEATHER)
    assert out.ok
    assert out.tool_calls == (ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"}),)
    assert out.text == ""


def test_parallel_calls_keep_order():
    raw = WEATHER + "\n" + WEATHER.replace("臺中市", "高雄市")
    out = parse_output(raw)
    assert [c.arguments["city"] for c in out.tool_calls] == ["臺中市", "高雄市"]


def test_text_around_calls_is_kept():
    out = parse_output("我幫你查一下。\n" + WEATHER)
    assert out.ok and out.text == "我幫你查一下。" and len(out.tool_calls) == 1


def test_think_block_is_ignored():
    out = parse_output("<think>\n使用者要天氣\n</think>\n\n" + WEATHER)
    assert out.ok and len(out.tool_calls) == 1 and out.text == ""


def test_prefilled_think_closed_by_model():
    out = parse_output("嗯，要查天氣。\n</think>\n\n" + WEATHER)
    assert out.ok and len(out.tool_calls) == 1 and out.text == ""


def test_arguments_as_json_string_are_accepted():
    raw = '<tool_call>{"name": "calendar_list", "arguments": "{\\"date\\": \\"2026-09-24\\"}"}</tool_call>'
    out = parse_output(raw)
    assert out.ok and out.tool_calls[0].arguments == {"date": "2026-09-24"}


def test_missing_arguments_defaults_to_empty():
    out = parse_output('<tool_call>{"name": "calendar_list"}</tool_call>')
    assert out.ok and out.tool_calls[0].arguments == {}


def test_invalid_json_is_an_error():
    out = parse_output('<tool_call>{"name": "get_weather", "arguments": {"city": }</tool_call>')
    assert not out.ok and "invalid JSON" in out.errors[0]


def test_truncated_output_is_an_error():
    out = parse_output('<tool_call>\n{"name": "get_weather", "arguments": {"city": "臺')
    assert not out.ok and "unclosed" in out.errors[0]


def test_missing_name_is_an_error():
    out = parse_output('<tool_call>{"arguments": {}}</tool_call>')
    assert not out.ok and "missing 'name'" in out.errors[0]


def test_one_bad_block_fails_the_whole_output():
    out = parse_output(WEATHER + "\n<tool_call>not json</tool_call>")
    assert not out.ok
    assert len(out.tool_calls) == 1  # good one still reported, for diagnostics


def test_stray_closing_tag_is_an_error():
    out = parse_output("好的</tool_call>")
    assert not out.ok


def test_render_round_trips():
    calls = (ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"}),)
    rendered = render_tool_calls(calls)
    assert rendered == WEATHER
    assert parse_output(rendered).tool_calls == calls
