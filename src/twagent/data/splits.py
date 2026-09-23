"""Group-wise train/valid/test split with a leakage guard (spec §5.3).

Every record of one scenario group -- (category, tools, style) -- goes to the
same split, and groups are assigned per category so each split sees every
category. The guard lives in `SplitResult.__post_init__`, as in
astro-vlm-lora: a leaky split object cannot be constructed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .dedup import matches_any
from .scenarios import Scenario


class DataLeakageError(ValueError):
    pass


def group_of(record: dict[str, Any]) -> str:
    return Scenario.from_dict(record["scenario"]).group


def assert_no_leakage(parts: dict[str, list[dict[str, Any]]], threshold: float) -> None:
    groups = {name: {group_of(r) for r in recs} for name, recs in parts.items()}
    for a, b in combinations(parts, 2):
        shared = groups[a] & groups[b]
        if shared:
            raise DataLeakageError(f"groups in both {a} and {b}: {sorted(shared)[:3]}")
        hits = matches_any(
            [r["message"] for r in parts[b]], [r["message"] for r in parts[a]], threshold
        )
        if hits:
            raise DataLeakageError(f"{len(hits)} {b} message(s) near-duplicate a {a} message")


@dataclass(frozen=True)
class SplitResult:
    train: list[dict[str, Any]]
    valid: list[dict[str, Any]]
    test: list[dict[str, Any]]
    threshold: float = 0.8

    def __post_init__(self) -> None:
        assert_no_leakage(
            {"train": self.train, "valid": self.valid, "test": self.test}, self.threshold
        )


def split_by_group(
    records: list[dict[str, Any]],
    *,
    seed: int,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    threshold: float = 0.8,
) -> SplitResult:
    by_category: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_category[r["scenario"]["category"]][group_of(r)].append(r)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for category in sorted(by_category):
        groups = by_category[category]
        names = sorted(groups)
        rng.shuffle(names)
        n = sum(len(v) for v in groups.values())
        n_test = n_valid = 0
        for name in names:
            recs = groups[name]
            if n_test < test_frac * n:
                test.extend(recs)
                n_test += len(recs)
            elif n_valid < valid_frac * n:
                valid.extend(recs)
                n_valid += len(recs)
            else:
                train.extend(recs)
    return SplitResult(train=train, valid=valid, test=test, threshold=threshold)
