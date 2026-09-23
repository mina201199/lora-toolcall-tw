"""raw trajectories -> filtered, deduplicated, split, formatted dataset.

    train.jsonl / valid.jsonl   rendered training examples (segments + loss mask)
    test_cases.jsonl            synthetic test set in the cases.py format
    data_report.json            counts, drop reasons, config

The synthetic test set is written as cases rather than trajectories: plan 3's
evaluation runs the model under test through `run_agent` and compares
against these expectations, exactly as it does for the handwritten set.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from twagent.cases import EXPECTED_DECISION, case_from_dict, validate_case
from twagent.render import render_chat

from .dedup import dedupe, matches_any
from .scenarios import Scenario
from .splits import SplitResult, group_of, split_by_group
from .validate import reason_key, validate_record


def to_training_example(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["scenario"]["id"],
        "group": group_of(record),
        "category": record["scenario"]["category"],
        "segments": [{"text": s.text, "train": s.train} for s in render_chat(record["messages"])],
    }


def to_case(record: dict[str, Any]) -> dict[str, Any]:
    s = Scenario.from_dict(record["scenario"])
    case = {
        "id": s.id,
        "category": s.category,
        "today": s.today.isoformat(),
        "messages": [{"role": "user", "content": record["message"]}],
        "expected": {
            "decision": EXPECTED_DECISION[s.category],
            "calls": record["result"]["tool_calls"]
            if EXPECTED_DECISION[s.category] == "call"
            else [],
        },
        "fail_tools": {s.tools[0]: s.failure} if s.failure else {},
        "note": f"synthetic; group={s.group}",
    }
    problems = validate_case(case_from_dict(case))
    if problems:
        raise ValueError(f"{s.id}: synthetic case is invalid: {problems}")
    return case


def build_dataset(
    raw: list[dict[str, Any]],
    *,
    handwritten_messages: list[str],
    seed: int,
    threshold: float = 0.8,
) -> tuple[SplitResult, dict[str, Any]]:
    reasons: Counter[str] = Counter()  # a rejected record may count under several
    kept = []
    for record in raw:
        problems = validate_record(record)
        if problems:
            reasons.update({reason_key(p) for p in problems})
        else:
            kept.append(record)
    rejected = len(raw) - len(kept)

    contaminated = set(matches_any([r["message"] for r in kept], handwritten_messages, threshold))
    kept = [r for i, r in enumerate(kept) if i not in contaminated]

    unique = dedupe([r["message"] for r in kept], threshold)
    near_duplicates = len(kept) - len(unique)
    kept = [kept[i] for i in unique]

    split = split_by_group(kept, seed=seed, threshold=threshold)
    report = {
        "raw": len(raw),
        "kept": len(kept),
        "keep_rate": round(len(kept) / len(raw), 4) if raw else 0.0,
        # raw == kept + rejected + contaminated + near_duplicates
        "rejected": rejected,
        "contaminated": len(contaminated),
        "near_duplicates": near_duplicates,
        "reject_reasons": dict(sorted(reasons.items())),
        "splits": {
            name: {
                "records": len(part),
                "groups": len({group_of(r) for r in part}),
                "by_category": dict(
                    sorted(Counter(r["scenario"]["category"] for r in part).items())
                ),
            }
            for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test))
        },
    }
    return split, report


def write_dataset(split: SplitResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "train.jsonl": [to_training_example(r) for r in split.train],
        "valid.jsonl": [to_training_example(r) for r in split.valid],
        "test_cases.jsonl": [to_case(r) for r in split.test],
    }
    paths = {}
    for name, items in rows.items():
        path = out_dir / name
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        paths[name] = path
    return paths
