"""`twagent synth` and `twagent build-dataset`.

    twagent synth --pilot 50                  # cost/keep-rate pilot -> data/synthetic/pilot.jsonl
    twagent synth --total 4000                # full run -> data/synthetic/raw.jsonl (resumable)
    twagent build-dataset                     # raw.jsonl -> train/valid/test_cases + report

The API key comes from the environment or a git-ignored `.env` file; this
code never prints or stores it. Spend is tracked in `<out>.cost.json` across
resumed runs -- checkpointed after every batch and on any exit, so a crash
does not lose spend -- and a run stops before its batch would start past
the budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
PROMPT_CACHE_KEY = "twagent-synth"  # stable across runs, so resumed runs hit the same cache


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


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    """Write via a temp file and `os.replace`, so a kill never leaves half a ledger."""
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class _NewRecords:
    """Counts records (complete lines) appended to a JSONL file since creation.

    Reads only the bytes added since the previous count, so checkpointing
    after every batch stays cheap on a large file. A partial last line is not
    counted and not skipped over: `synthesize` truncates it away and appends
    from exactly that offset.
    """

    def __init__(self, path: Path) -> None:
        self.path, self.offset, self.lines = path, 0, 0
        self.base = self._advance()

    def _advance(self) -> int:
        if self.path.exists():
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
            self.lines += data.count(b"\n")
            self.offset += data.rfind(b"\n") + 1
        return self.lines

    def count(self) -> int:
        return self._advance() - self.base


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

    common = {"reasoning_effort": cfg.reasoning_effort, "prompt_cache_key": PROMPT_CACHE_KEY}
    teacher = GPTClient(model, max_tokens=cfg.max_tokens, **common)
    query = GPTClient(model, max_tokens=cfg.max_tokens * 4, **common)
    ledger_path = out.with_name(out.name + ".cost.json")
    ledger = _read_ledger(ledger_path)
    previous = ledger["spent_usd"]
    run: dict[str, Any] = {"model": model}
    ledger["runs"].append(run)  # updated in place at every checkpoint
    new_records = _NewRecords(out)

    def spent() -> float:
        return previous + teacher.usage.cost_usd(price) + query.usage.cost_usd(price)

    def checkpoint() -> None:
        ledger["spent_usd"] = round(spent(), 6)
        run["written"] = new_records.count()
        run["cost_usd"] = round(spent() - previous, 6)
        run["teacher_usage"] = dict(vars(teacher.usage))
        run["query_usage"] = dict(vars(query.usage))
        _write_ledger(ledger_path, ledger)

    try:
        report = synthesize(
            scenarios,
            query_llm=query,
            teacher=teacher,
            out_path=out,
            batch_size=cfg.query_batch_size,
            spent_usd=spent,
            budget_usd=budget,
            on_progress=checkpoint,
        )
    finally:
        checkpoint()  # an exception or Ctrl-C still records what was spent

    if report.repaired_bytes:
        print(f"WARNING repaired {out}: dropped a partial last line ({report.repaired_bytes} "
              "bytes) left by an interrupted run; its scenario counts as not done")  # fmt: skip
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
        verdict = "within budget" if projected <= budget else "EXCEEDS budget"
        print(f"projection for train_target={cfg.train_target}: {needed} raw records, "
              f"~${projected:.2f} with {model} ({verdict} ${budget:.2f})")  # fmt: skip
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    raw = load_records(Path(args.raw))
    if not raw:
        print(f"ERROR no records in {args.raw}")
        return 1
    handwritten: list[str] = []
    if Path(args.handwritten).exists():
        cases, errors = load_cases(args.handwritten)
        handwritten = [c.messages[-1]["content"] for c in cases]
        if errors:
            # Count cases, not messages (one case can have several problems), and
            # never echo the errors: they quote the user's hand-written cases.
            invalid = len({re.match(r"line \d+", e)[0] for e in errors})
            print(f"WARNING handwritten set has {invalid} invalid case(s); "
                  "contamination check uses the valid ones only")  # fmt: skip
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
