"""The one system prompt, rendered by us rather than by the chat template.

We render Qwen's tool block ourselves and send it as plain system text, with
no `tools` field in the request. Two reasons:

1. The server returns the raw completion, so `parsing.py` -- not the server --
   decides what counts as a tool call. That is what lets us measure parse
   failures at all.
2. Training data (next plan) is formatted with this same function, so the
   prompt the model is trained on and the prompt it is served with cannot
   drift apart.

`scripts/check_template.py` verifies that this rendering is byte-identical to
what the model's own chat template produces when given `tools`.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from twagent.dates import format_today, invoice_period, previous_invoice_period
from twagent.tools.schemas import TOOLS

PERSONA = (
    "你是臺灣生活助理，可以查天氣、高鐵與臺鐵班次、YouBike 站點、統一發票對獎，並管理使用者的行事曆。\n"
    "- 一律使用繁體中文與臺灣用語回覆。\n"
    "- 今天是 {today}，時區為 Asia/Taipei。所有相對日期都以今天為準換算。\n"
    "- 統一發票「這一期」是 {invoice}，「上一期」是 {previous_invoice}。\n"
    "- 必填資訊不足時，先向使用者追問，不要自行猜測；選填參數（例如出發時間、車種、地點）"
    "使用者沒提到就省略，直接查詢。\n"
    "- 需要查詢或新增時，直接輸出工具呼叫，不要只說「我來查」。\n"
    "- 超出工具能力範圍的請求（例如訂票、付款），直接說明做不到。"
)


def render_tools_block(tools: list[dict[str, Any]]) -> str:
    """Qwen's tool instructions, character for character.

    Mirrors the `if tools` branch of the Qwen chat template, where each tool
    goes through `tojson` (json.dumps with ensure_ascii=False).
    """
    lines = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
    return (
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        f"{lines}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def build_persona(today: date) -> str:
    return PERSONA.format(
        today=format_today(today),
        invoice=invoice_period(today),
        previous_invoice=previous_invoice_period(today),
    )


def build_system_prompt(today: date, tools: list[dict[str, Any]] | None = None) -> str:
    return build_persona(today) + "\n\n" + render_tools_block(TOOLS if tools is None else tools)
