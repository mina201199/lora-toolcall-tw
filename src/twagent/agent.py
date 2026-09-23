"""The only agent loop: model -> tools -> model, until the model answers.

Evaluation and the server both call `run_agent`. Failure policy:

- malformed <tool_call>   -> tell the model, let it retry (`parse_retries`)
- schema-invalid call     -> return the validation errors as the tool result
                             so the model can fix them (`schema_retries`)
- tool raised ToolError   -> return the error as the tool result; never
                             substitute another executor's data
- LLM unreachable, or a retry budget is exhausted
                          -> switch to `fallback` if one was given, else stop
                             with a fixed apology
- more than `max_steps` LLM calls -> stop
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from twagent.config import AgentConfig
from twagent.llm import LLMClient, LLMError, Message
from twagent.parsing import ToolCall, parse_output, render_tool_calls
from twagent.tools.base import ToolError, ToolExecutor
from twagent.tools.schemas import validate_arguments

FAILURE_MESSAGE = "抱歉，我這次沒辦法正確處理你的請求，請換個說法再試一次。"
MAX_STEPS_MESSAGE = "抱歉，這個請求需要的步驟太多，我先停在這裡。請把需求拆小一點再試試。"
PARSE_RETRY_PROMPT = "（系統訊息）你上一則輸出的工具呼叫格式錯誤：{errors}。請重新輸出，工具呼叫必須是 <tool_call> 內的合法 JSON。"


@dataclass(frozen=True)
class TraceEvent:
    type: str  # llm_output | parse_error | schema_error | tool_result | tool_error | llm_error | fallback
    data: dict[str, Any]


@dataclass
class AgentResult:
    final_text: str
    stopped_reason: str  # answer | max_steps | parse_failure | schema_failure | llm_error
    messages: list[Message]
    trace: list[TraceEvent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)  # valid calls, in execution order
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "stopped_reason": self.stopped_reason,
            "used_fallback": self.used_fallback,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "trace": [{"type": e.type, **e.data} for e in self.trace],
        }


def _assistant_content(text: str, calls: tuple[ToolCall, ...]) -> str:
    rendered = render_tool_calls(calls)
    return f"{text}\n{rendered}" if text else rendered


def run_agent(
    conversation: list[Message],
    *,
    llm: LLMClient,
    tools: ToolExecutor,
    system_prompt: str,
    config: AgentConfig,
    fallback: LLMClient | None = None,
) -> AgentResult:
    messages: list[Message] = [{"role": "system", "content": system_prompt}, *conversation]
    result = AgentResult(final_text="", stopped_reason="", messages=messages)
    active = llm
    parse_failures = 0
    schema_failures = 0
    retry_start: int | None = None  # where the current parse-retry exchange began

    def escalate(reason: str) -> bool:
        """Switch to the fallback model. False means there is none left."""
        nonlocal active, parse_failures, schema_failures, retry_start
        if fallback is None or active is fallback:
            result.final_text, result.stopped_reason = FAILURE_MESSAGE, reason
            return False
        if retry_start is not None:
            del messages[retry_start:]  # the fallback should not see the garbage
        active, parse_failures, schema_failures, retry_start = fallback, 0, 0, None
        result.used_fallback = True
        result.trace.append(TraceEvent("fallback", {"reason": reason}))
        return True

    for _ in range(config.max_steps):
        try:
            raw = active.complete(messages)
        except LLMError as e:
            result.trace.append(TraceEvent("llm_error", {"error": str(e)}))
            if escalate("llm_error"):
                continue
            return result
        result.trace.append(TraceEvent("llm_output", {"raw": raw}))
        parsed = parse_output(raw)

        if not parsed.ok:
            result.trace.append(TraceEvent("parse_error", {"errors": list(parsed.errors)}))
            if parse_failures < config.parse_retries:
                parse_failures += 1
                if retry_start is None:
                    retry_start = len(messages)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": PARSE_RETRY_PROMPT.format(errors="；".join(parsed.errors)),
                    }
                )
                continue
            if escalate("parse_failure"):
                continue
            return result
        retry_start = None

        if not parsed.tool_calls:
            messages.append({"role": "assistant", "content": parsed.text})
            result.final_text, result.stopped_reason = parsed.text, "answer"
            return result

        messages.append(
            {"role": "assistant", "content": _assistant_content(parsed.text, parsed.tool_calls)}
        )
        # Validate every call in the turn before executing any of them: running
        # half of what the model meant (the valid calls) while the rest is
        # rejected is worse than asking the whole batch to be resent, and it
        # would double-execute the valid calls once the model corrects and
        # resends the turn.
        validated = [
            (call, validate_arguments(call.name, call.arguments)) for call in parsed.tool_calls
        ]
        any_invalid = any(errors for _, errors in validated)
        for call, errors in validated:
            if errors:
                result.trace.append(
                    TraceEvent("schema_error", {"call": call.to_dict(), "errors": errors})
                )
                payload: dict[str, Any] = {"error": "參數驗證失敗", "details": errors}
            elif any_invalid:
                payload = {"error": "未執行：同一輪有其他工具呼叫的參數錯誤，請修正後整批重新送出"}
            else:
                try:
                    payload = tools.call(call.name, call.arguments)
                    result.tool_calls.append(call)
                    result.trace.append(
                        TraceEvent("tool_result", {"call": call.to_dict(), "result": payload})
                    )
                except ToolError as e:
                    result.tool_calls.append(call)
                    result.trace.append(
                        TraceEvent("tool_error", {"call": call.to_dict(), "error": str(e)})
                    )
                    payload = {"error": str(e)}
            messages.append({"role": "tool", "content": json.dumps(payload, ensure_ascii=False)})

        if any_invalid:
            schema_failures += 1
            if schema_failures > config.schema_retries and not escalate("schema_failure"):
                return result

    result.final_text, result.stopped_reason = MAX_STEPS_MESSAGE, "max_steps"
    return result
