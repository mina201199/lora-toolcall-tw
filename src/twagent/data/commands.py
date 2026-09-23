"""`twagent synth` and `twagent build-dataset`.

    twagent synth --pilot 50                  # cost/keep-rate pilot -> data/synthetic/pilot.jsonl
    twagent synth --total 4000                # full run -> data/synthetic/raw.jsonl (resumable)
    twagent build-dataset                     # raw.jsonl -> train/valid/test_cases + report

The API key comes from the environment or a git-ignored `.env` file; this
code never prints or stores it. Spend is tracked in `<out>.cost.json` across
resumed runs, and a run stops before its batch would start past the budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from twagent.cases import load_cases
from twagent.config import load_config
from twagent.gpt import GPTClient

from .build import build_dataset, write_dataset
from .scenarios import build_scenarios
from .synth import load_records, synthesize
from .validate import validate_record

TRAIN_SHARE = 0.8  # split_by_group puts ~10% in valid and ~10% in test


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader; never overrides a variable already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _read_ledger(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"spent_usd": 0.0, "runs": []}


def cmd_synth(args: argparse.Namespace) -> int:
    cfg = load_config(args.config).synth
    load_dotenv(Path(".env"))
    model = args.model or cfg.model
    if model not in cfg.prices:
        print(f"ERROR no price for {model!r} in synth.prices; add it before spending money")
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR OPENAI_API_KEY is not set; put it in .env or the environment")
        return 2
    price = cfg.prices[model]
    if args.pilot:
        scenarios = build_scenarios(args.pilot, seed=cfg.seed + 1)
        out = Path(args.out or "data/synthetic/pilot.jsonl")
    else:
        scenarios = build_scenarios(args.total, seed=cfg.seed)
        out = Path(args.out or "data/synthetic/raw.jsonl")
    budget = args.budget if args.budget is not None else cfg.budget_usd

    teacher = GPTClient(model, max_tokens=cfg.max_tokens, reasoning_effort=cfg.reasoning_effort)
    query = GPTClient(model, max_tokens=cfg.max_tokens * 4, reasoning_effort=cfg.reasoning_effort)
    ledger_path = out.with_name(out.name + ".cost.json")
    ledger = _read_ledger(ledger_path)
    previous = ledger["spent_usd"]

    def spent() -> float:
        return previous + teacher.usage.cost_usd(price) + query.usage.cost_usd(price)

    report = synthesize(
        scenarios,
        query_llm=query,
        teacher=teacher,
        out_path=out,
        batch_size=cfg.query_batch_size,
        spent_usd=spent,
        budget_usd=budget,
    )
    ledger["spent_usd"] = round(spent(), 6)
    ledger["runs"].append(
        {
            "model": model,
            "written": report.written,
            "cost_usd": round(spent() - previous, 6),
            "teacher_usage": vars(teacher.usage),
            "query_usage": vars(query.usage),
        }
    )
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")

    records = load_records(out)
    kept = sum(1 for r in records if not validate_record(r))
    keep_rate = kept / len(records) if records else 0.0
    per_record = ledger["spent_usd"] / len(records) if records else 0.0
    print(f"model {model} | wrote {report.written} | already had {report.already_done} | "
          f"missing queries {report.missing_queries} | llm errors {report.llm_errors} | "
          f"stopped: {report.stopped}")  # fmt: skip
    print(f"records {len(records)} | keep rate {keep_rate:.1%} | spent ${ledger['spent_usd']:.2f} "
          f"| ${per_record:.4f} per record")  # fmt: skip
    if keep_rate > 0:
        needed = math.ceil(cfg.train_target / TRAIN_SHARE / keep_rate)
        projected = needed * per_record
        verdict = "within budget" if projected <= cfg.budget_usd else "EXCEEDS budget"
        print(f"projection for train_target={cfg.train_target}: {needed} raw records, "
              f"~${projected:.2f} with {model} ({verdict} ${cfg.budget_usd:.2f})")  # fmt: skip
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    raw = load_records(Path(args.raw))
    if not raw:
        print(f"ERROR no records in {args.raw}")
        return 1
    handwritten: list[str] = []
    if Path(args.handwritten).exists():
        cases, _ = load_cases(args.handwritten)
        handwritten = [c.messages[-1]["content"] for c in cases]
    split, report = build_dataset(raw, handwritten_messages=handwritten, seed=cfg.synth.seed)
    out_dir = Path(args.out_dir)
    write_dataset(split, out_dir)
    report["handwritten_reference"] = len(handwritten)
    report["config"] = cfg.to_dict()
    (out_dir / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"raw {report['raw']} -> kept {report['kept']} ({report['keep_rate']:.1%}); "
          f"rejected {report['rejected']}, contaminated {report['contaminated']}, "
          f"near-duplicates {report['near_duplicates']}")  # fmt: skip
    for name, part in report["splits"].items():
        print(f"  {name}: {part['records']} records, {part['groups']} groups")
    return 0


def add_parsers(sub: Any, default_config: Path) -> None:
    p = sub.add_parser("synth", help="generate teacher trajectories with an OpenAI model")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", type=int, help="run N pilot scenarios (separate seed and file)")
    mode.add_argument("--total", type=int, help="scenario count for the full run")
    p.add_argument("--model", help="override synth.model")
    p.add_argument("--budget", type=float, help="override synth.budget_usd for this run")
    p.add_argument("--out", help="output JSONL")
    p.add_argument("--config", type=Path, default=default_config)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("build-dataset", help="validate, dedupe, split and format raw records")
    p.add_argument("--raw", default="data/synthetic/raw.jsonl")
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--handwritten", default="data/handwritten_test.jsonl")
    p.add_argument("--config", type=Path, default=default_config)
    p.set_defaults(func=cmd_build_dataset)
