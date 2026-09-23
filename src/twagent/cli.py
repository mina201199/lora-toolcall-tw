"""`twagent` command line.

twagent check-cases data/handwritten_test.jsonl
twagent chat "明天台中會下雨嗎" [--today 2026-09-23] [--base-url URL] [--json]
twagent synth --pilot 50 | --total N      (see twagent.data.commands)
twagent build-dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from twagent.agent import run_agent
from twagent.cases import load_cases
from twagent.config import load_config
from twagent.data import commands as data_commands
from twagent.llm import OpenAICompatClient
from twagent.prompting import build_system_prompt
from twagent.tools.mock import MockTools

DEFAULT_CONFIG = Path("configs/default.yaml")
TAIPEI = timezone(timedelta(hours=8))


def _check_cases(args: argparse.Namespace) -> int:
    cases, errors = load_cases(args.path)
    for e in errors:
        print(f"ERROR {e}")
    counts = Counter(c.category for c in cases)
    print(
        f"{len(cases)} valid case(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return 1 if errors else 0


def _chat(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    llm = OpenAICompatClient(
        args.base_url or cfg.llm.base_url,
        cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout_s=cfg.llm.timeout_s,
    )
    today = date.fromisoformat(args.today) if args.today else datetime.now(TAIPEI).date()
    result = run_agent(
        [{"role": "user", "content": args.message}],
        llm=llm,
        tools=MockTools(),
        system_prompt=build_system_prompt(today),
        config=cfg.agent,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    for event in result.trace:
        print(f"[{event.type}] {json.dumps(event.data, ensure_ascii=False)}")
    print(f"\n({result.stopped_reason}) {result.final_text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles often default to a legacy code page (e.g. cp950) that
    # cannot encode most of what this CLI prints (Traditional Chinese, emoji).
    # Force UTF-8 on stdout/stderr where the stream supports it (redirected
    # output and non-Windows platforms may not have `reconfigure`).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="twagent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-cases", help="validate a test-case JSONL file")
    p.add_argument("path", type=Path)
    p.set_defaults(func=_check_cases)

    p = sub.add_parser("chat", help="run one message through the agent (mock tools)")
    p.add_argument("message")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--today", help="YYYY-MM-DD; defaults to the real date")
    p.add_argument("--base-url", help="override llm.base_url")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.set_defaults(func=_chat)

    data_commands.add_parsers(sub, DEFAULT_CONFIG)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
