"""Two-stage synthesis (spec §5.2, as revised for plan 2):

1. A generator model writes one user message per scenario (batched).
2. A teacher model runs through the real agent loop -- `run_agent` with
   `MockTools` and the exact Qwen system prompt -- and its trajectory becomes
   the training sample. Tool results always come from the mocks, never from
   the model, and a sequential step naturally sees the previous result.

The teacher gets no second chances (`parse_retries=0`, `schema_retries=0`):
a trajectory with any error is recorded and later dropped by validation, so
the training data never shows a model recovering from its own bad output.
The exception is an API failure (`stopped_reason == "llm_error"`): it says
nothing about the scenario, so it is not recorded and a rerun retries it.
`MAX_CONSECUTIVE_LLM_ERRORS` failures in a row stop the run, so a persistent
error (a bad request, an outage) does not burn through the whole matrix.

Output is append-only JSONL, one record per scenario, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twagent.agent import run_agent
from twagent.config import AgentConfig
from twagent.dates import format_today
from twagent.llm import LLMClient, LLMError
from twagent.prompting import build_system_prompt
from twagent.tools.mock import MockTools
from twagent.tools.schemas import get_tool

from .scenarios import Scenario

TEACHER_CONFIG = AgentConfig(max_steps=5, parse_retries=0, schema_retries=0)
MAX_CONSECUTIVE_LLM_ERRORS = 3

_CATEGORY_BRIEF = {
    "single": "只需要呼叫一次下列工具就能完成，而且該工具所需的資訊都已經說清楚。",
    "parallel": "一句話裡同時要兩件互不相依的事，分別對應下列工具（同一個工具出現兩次代表兩組不同參數，例如兩個城市）。資訊都要齊全。",
    "sequential": "必須先用第一個工具查到結果，再根據查到的內容使用第二個工具（例如查到班次後，把搭乘時間排進行事曆；看了天氣後決定要不要排行程）。資訊都要齊全。",
    "clarify": "想用下列工具，但刻意漏掉至少一個必要資訊（例如沒說起訖站、沒說日期、沒給發票號碼、沒說行程時間），讓助理必須先追問。",
    "no_tool": "完全不需要任何工具。類型：{kind}。",
    "tool_error": "一般的單一工具需求，資訊齊全（使用者不知道工具之後會故障）。",
}

_RULES = """規則：
- 使用繁體中文與臺灣用語，像真人在手機上隨手打字；依指定的風格寫。
- 不可以出現工具的英文名稱，也不要提到「工具」「API」。
- 地名、車站可以用口語說法（台北、高雄、台大附近）。
- 有指定「日期說法」時，必須原封不動使用那個說法，不要自己寫出具體日期。
- 發票對獎要附上 8 碼號碼，並用「這一期」「上一期」或月份描述期別。
- 新增行程要講清楚事由與時間（開始與結束，或開始與持續多久）。
- 每則 5 到 80 個字；風格是「很長而且繞圈子」時可以到 150 字。
- 只輸出一個 JSON 陣列，不要任何其他文字。每個元素為 {"id": "...", "message": "..."}，順序與清單相同。"""


def _describe(s: Scenario) -> str:
    brief = _CATEGORY_BRIEF[s.category].format(kind=s.kind)
    parts = [f"id：{s.id}", f"需求：{brief}", f"風格：{s.style}", f"今天：{format_today(s.today)}"]
    if s.tools:
        tools = "、".join(f"{t}（{get_tool(t)['function']['description']}）" for t in s.tools)
        parts.append(f"工具：{tools}")
    if s.date_expr:
        parts.append(f"日期說法：必須使用「{s.date_expr}」")
    return "- " + "｜".join(parts)


def build_query_prompt(batch: list[Scenario]) -> str:
    lines = "\n".join(_describe(s) for s in batch)
    return (
        "請替一個臺灣生活助理的訓練資料，寫出使用者傳給助理的第一則訊息。"
        "每個情境寫一則。\n\n"
        f"{_RULES}\n\n情境清單：\n{lines}"
    )


def parse_query_response(raw: str, batch: list[Scenario]) -> dict[str, str]:
    """Scenario id -> message. Anything malformed or unknown is left out."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return {}
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    wanted = {s.id for s in batch}
    out: dict[str, str] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        sid, msg = item.get("id"), item.get("message")
        if sid in wanted and isinstance(msg, str) and msg.strip():
            out[sid] = msg.strip()
    return out


def run_trajectory(scenario: Scenario, message: str, teacher: LLMClient) -> dict[str, Any]:
    failures = {scenario.tools[0]: scenario.failure} if scenario.failure else None
    tools = MockTools(failures=failures)
    result = run_agent(
        [{"role": "user", "content": message}],
        llm=teacher,
        tools=tools,
        system_prompt=build_system_prompt(scenario.today),
        config=TEACHER_CONFIG,
    )
    return {
        "scenario": scenario.to_dict(),
        "message": message,
        "result": result.to_dict(),
        "messages": result.messages,
        "calendar": tools.events(),
    }


@dataclass
class SynthReport:
    written: int = 0
    already_done: int = 0
    missing_queries: int = 0
    llm_errors: int = 0  # teacher trajectories lost to an API error (not written)
    stopped: str = "complete"  # complete | budget | llm_error


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _chunks(items: list[Scenario], size: int) -> Iterable[list[Scenario]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def synthesize(
    scenarios: list[Scenario],
    *,
    query_llm: LLMClient,
    teacher: LLMClient,
    out_path: Path,
    batch_size: int,
    spent_usd: Callable[[], float],
    budget_usd: float,
) -> SynthReport:
    """Append one record per scenario to `out_path`.

    Stops before a batch would start at or past the budget, or after
    `MAX_CONSECUTIVE_LLM_ERRORS` LLM failures in a row.
    """
    done = {r["scenario"]["id"] for r in load_records(out_path)}
    report = SynthReport(already_done=len(done))
    todo = [s for s in scenarios if s.id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0  # consecutive LLM failures: query calls or teacher trajectories
    with open(out_path, "a", encoding="utf-8") as f:
        for batch in _chunks(todo, batch_size):
            if spent_usd() >= budget_usd:
                report.stopped = "budget"
                break
            try:
                raw = query_llm.complete([{"role": "user", "content": build_query_prompt(batch)}])
            except LLMError:
                report.missing_queries += len(batch)
                failures += 1
            else:
                failures = 0
                messages = parse_query_response(raw, batch)
                for s in batch:
                    if s.id not in messages:
                        report.missing_queries += 1
                        continue
                    record = run_trajectory(s, messages[s.id], teacher)
                    if record["result"]["stopped_reason"] == "llm_error":
                        report.llm_errors += 1  # not written: a rerun retries it
                        failures += 1
                        if failures >= MAX_CONSECUTIVE_LLM_ERRORS:
                            break
                        continue
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    report.written += 1
                    failures = 0
            if failures >= MAX_CONSECUTIVE_LLM_ERRORS:
                report.stopped = "llm_error"
                break
    return report
