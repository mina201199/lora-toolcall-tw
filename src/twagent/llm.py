"""LLM clients. Every client takes chat messages and returns raw text.

Returning raw text (not structured tool calls) is deliberate: the agent runs
every model's output through the same `parse_output`, so a parse failure is
something we can count, whichever model produced it.

Messages are plain dicts with `role` in {system, user, assistant, tool} and a
string `content`. Assistant tool calls live inside `content` in their
canonical `<tool_call>` text form (see `parsing.render_tool_calls`).
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

Message = dict[str, str]


class LLMError(Exception):
    """The model could not be reached or returned something unusable."""


class LLMClient(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class OpenAICompatClient:
    """Client for any OpenAI-compatible `/chat/completions` endpoint.

    Used for llama-server. Never sends `tools`; the tool block is already in
    the system prompt (see `prompting.py`).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 60.0,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Qwen3 hybrid models think by default; tool calling does not want it.
        # Harmless for templates that ignore the flag.
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.extra_body.update(extra_body or {})
        self._http = httpx.Client(timeout=timeout_s, transport=transport)

    def complete(self, messages: list[Message]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            **self.extra_body,
        }
        try:
            resp = self._http.post(self.url, json=body)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"request to {self.url} failed: {e}") from e
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {data!r:.200}") from e
        return content or ""


class ScriptedLLM:
    """Test double: returns canned responses in order and records its inputs."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)
