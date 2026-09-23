"""The only code that turns model text into tool calls.

Evaluation and the server both call `parse_output`. A second parser would be
the same mistake as astro-vlm-lora's BUG-1: two copies of one operation, and
the published number comes from the wrong one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ParsedOutput:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    errors: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_block(index: int, body: str) -> tuple[ToolCall | None, str | None]:
    try:
        obj = json.loads(body.strip())
    except json.JSONDecodeError as e:
        return None, f"tool_call #{index}: invalid JSON ({e.msg})"
    if not isinstance(obj, dict):
        return None, f"tool_call #{index}: expected an object"
    name, args = obj.get("name"), obj.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None, f"tool_call #{index}: missing 'name'"
    if isinstance(args, str):
        # Some models emit arguments as a JSON-encoded string.
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None, f"tool_call #{index}: 'arguments' is not valid JSON"
    if not isinstance(args, dict):
        return None, f"tool_call #{index}: 'arguments' must be an object"
    return ToolCall(name=name, arguments=args), None


def parse_output(raw: str) -> ParsedOutput:
    """Split a completion into free text and tool calls.

    Any malformed block makes the whole output not-ok: executing half of what
    the model meant is worse than asking it again.
    """
    s = _THINK.sub("", raw)
    if "</think>" in s:  # template pre-filled <think>, model closed it
        s = s.split("</think>", 1)[1]
    calls: list[ToolCall] = []
    errors: list[str] = []
    for i, m in enumerate(_BLOCK.finditer(s), start=1):
        call, err = _parse_block(i, m.group(1))
        if call:
            calls.append(call)
        else:
            errors.append(err)
    rest = _BLOCK.sub("", s)
    if "<tool_call>" in rest:
        errors.append("unclosed <tool_call> (output truncated?)")
    elif "</tool_call>" in rest:
        errors.append("</tool_call> without matching <tool_call>")
    text = rest.replace("<tool_call>", "").strip()
    return ParsedOutput(text=text, tool_calls=tuple(calls), errors=tuple(errors))


def render_tool_calls(calls: list[ToolCall] | tuple[ToolCall, ...]) -> str:
    """Canonical text form, identical to how the Qwen template renders
    an assistant message's `tool_calls`."""
    return "\n".join(
        "<tool_call>\n" + json.dumps(c.to_dict(), ensure_ascii=False) + "\n</tool_call>"
        for c in calls
    )
