"""Near-duplicate detection on user messages (MinHash over character 3-grams).

Punctuation and whitespace are ignored, so "明天台中會下雨嗎？" and
"明天台中會下雨嗎" are the same message. LSH proposes candidates; each is
confirmed with the MinHash Jaccard estimate before it counts.
"""

from __future__ import annotations

import unicodedata

from datasketch import MinHash, MinHashLSH

NUM_PERM = 128


def _normalise(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "Z", "C")))


def _shingles(text: str, n: int = 3) -> set[str]:
    t = _normalise(text)
    if len(t) <= n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM, seed=1)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def dedupe(texts: list[str], threshold: float) -> list[int]:
    """Indices to keep: the first of every near-duplicate cluster survives."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    hashes: dict[str, MinHash] = {}
    keep: list[int] = []
    for i, text in enumerate(texts):
        m = minhash(text)
        if any(m.jaccard(hashes[k]) >= threshold for k in lsh.query(m)):
            continue
        key = str(i)
        lsh.insert(key, m)
        hashes[key] = m
        keep.append(i)
    return keep


def matches_any(texts: list[str], reference: list[str], threshold: float) -> list[int]:
    """Indices of `texts` that are near-duplicates of any `reference` text."""
    if not texts or not reference:
        return []
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    hashes: dict[str, MinHash] = {}
    for i, text in enumerate(reference):
        key = f"r{i}"
        hashes[key] = minhash(text)
        lsh.insert(key, hashes[key])
    hits = []
    for i, text in enumerate(texts):
        m = minhash(text)
        if any(m.jaccard(hashes[k]) >= threshold for k in lsh.query(m)):
            hits.append(i)
    return hits
