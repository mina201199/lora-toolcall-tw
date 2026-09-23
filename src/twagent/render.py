"""Qwen chat-template rendering, done by us, with loss-mask boundaries.

Training (plan 3) tokenises the segments this module returns instead of calling
`tokenizer.apply_chat_template`, so the text the model is trained on is exactly
the text `tests/fixtures/qwen_*.txt` recorded from the model's own template --
the tests below the module prove it byte for byte.

Only assistant content (plus its closing `<|im_end|>`) is marked `train=True`.
System, user and tool-result text is context: the model should read it, never
learn to produce it.

Consecutive `tool` messages are merged into one user turn, each wrapped in
`<tool_response>` -- that is how the Qwen template renders them.
"""

from __future__ import annotations

from dataclasses import dataclass

from twagent.llm import Message

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass(frozen=True)
class Segment:
    text: str
    train: bool


def render_chat(messages: list[Message], *, add_generation_prompt: bool = False) -> list[Segment]:
    segments: list[Segment] = []
    i = 0
    while i < len(messages):
        role, content = messages[i]["role"], messages[i]["content"]
        if role in ("system", "user"):
            segments.append(Segment(f"{IM_START}{role}\n{content}{IM_END}\n", train=False))
        elif role == "assistant":
            segments.append(Segment(f"{IM_START}assistant\n", train=False))
            segments.append(Segment(f"{content}{IM_END}", train=True))
            segments.append(Segment("\n", train=False))
        elif role == "tool":
            responses = []
            while i < len(messages) and messages[i]["role"] == "tool":
                responses.append(f"\n<tool_response>\n{messages[i]['content']}\n</tool_response>")
                i += 1
            segments.append(Segment(f"{IM_START}user{''.join(responses)}{IM_END}\n", train=False))
            continue
        else:
            raise ValueError(f"unknown role {role!r}")
        i += 1
    if add_generation_prompt:
        segments.append(Segment(f"{IM_START}assistant\n", train=False))
    return segments


def to_text(segments: list[Segment]) -> str:
    return "".join(s.text for s in segments)
