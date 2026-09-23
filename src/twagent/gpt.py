"""OpenAI GPT as an `LLMClient`: the data-synthesis teacher, and later the B2
baseline and the serving fallback.

GPT does not get OpenAI `tools`. It gets the same system prompt Qwen gets --
which already contains the `<tool_call>` instructions -- and answers in plain
text, so its output goes through the same `parse_output` as every other model.
Tool results are sent back the way the Qwen template renders them:
`<tool_response>` blocks inside a user turn.

For prompt caching the system prompt is split into two system messages: the
tool block (large, identical for every request) goes first and ends with an
explicit `prompt_cache_breakpoint`; the persona (small, contains today's date)
follows it. OpenAI's default implicit breakpoint stays on, so later turns of
one trajectory reuse the earlier turns as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import openai

from twagent.config import ModelPrice
from twagent.llm import LLMError, Message

TOOLS_MARKER = "\n\n# Tools\n"


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0  # neither read from nor written to the cache
    output_tokens: int = 0  # includes reasoning tokens
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0  # informational; already counted in output_tokens

    def add(self, usage: Any) -> None:
        self.requests += 1
        if usage is None:
            return
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        written = getattr(details, "cache_write_tokens", 0) or 0
        # prompt_tokens counts every input token, cached and written ones included.
        self.input_tokens += max(0, (usage.prompt_tokens or 0) - cached - written)
        self.cached_tokens += cached
        self.cache_write_tokens += written
        self.output_tokens += usage.completion_tokens or 0
        out_details = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens += getattr(out_details, "reasoning_tokens", 0) or 0

    def cost_usd(self, price: ModelPrice) -> float:
        return (
            self.input_tokens * price.input
            + self.output_tokens * price.output
            + self.cached_tokens * price.cache_read
            + self.cache_write_tokens * price.cache_write
        ) / 1_000_000


def _system_messages(system: str) -> list[dict[str, Any]]:
    if TOOLS_MARKER not in system:
        return [{"role": "system", "content": system}]
    persona, tools = system.split(TOOLS_MARKER, 1)
    tools_part = {
        "type": "text",
        "text": "# Tools\n" + tools,
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    return [
        {"role": "system", "content": [tools_part]},
        {"role": "system", "content": persona},
    ]


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Our messages -> Chat Completions messages.

    Tool results become `<tool_response>` text in a user turn, and consecutive
    same-role turns are merged, matching how the Qwen template renders them.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            out.extend(_system_messages(content))
            continue
        if role == "tool":
            role, content = "user", f"<tool_response>\n{content}\n</tool_response>"
        elif role not in ("user", "assistant"):
            raise ValueError(f"unknown role {role!r}")
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n" + content
        else:
            out.append({"role": role, "content": content})
    return out


class GPTClient:
    """`LLMClient` backed by Chat Completions. `usage` accumulates every call."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 1024,
        reasoning_effort: str = "none",
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.usage = Usage()
        # Reads OPENAI_API_KEY from the environment; the key never passes through here.
        self._client = client if client is not None else openai.OpenAI(max_retries=3, timeout=120)

    def complete(self, messages: list[Message]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as e:
            raise LLMError(f"{self.model}: {e}") from e
        self.usage.add(response.usage)
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"{self.model} refused the request")
        if choice.finish_reason == "length":
            raise LLMError(f"{self.model} hit max_tokens={self.max_tokens}")
        if choice.finish_reason == "content_filter":
            raise LLMError(f"{self.model} output was filtered")
        return choice.message.content or ""
