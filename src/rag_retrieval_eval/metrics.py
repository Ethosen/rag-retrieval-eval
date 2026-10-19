"""Ranking metrics for the case where each query has exactly one relevant section."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .data import Query
from .retrievers import Ranking

CUTOFFS = (1, 5, 10, 20)


def rank_of(target: str, ranked_keys: Sequence[str]) -> int | None:
    """1-based rank of the target, or None when it was not retrieved."""
    try:
        return ranked_keys.index(target) + 1
    except ValueError:
        return None


def per_query(query: Query, ranking: Ranking) -> dict[str, float]:
    keys = [key for key, _ in ranking]
    docs = list(dict.fromkeys(key.split("#", 1)[0] for key in keys))
    rank = rank_of(query.section_key, keys)
    doc_rank = rank_of(query.doc_id, docs)
    row = {f"recall@{c}": float(rank is not None and rank <= c) for c in CUTOFFS}
    row["mrr@10"] = 1.0 / rank if rank is not None and rank <= 10 else 0.0
    # With one relevant item the ideal DCG is 1, so nDCG is just the discounted gain at its rank.
    row["ndcg@10"] = 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0
    row["doc_recall@5"] = float(doc_rank is not None and doc_rank <= 5)
    return row


def evaluate(queries: Sequence[Query], rankings: Sequence[Ranking]) -> list[dict[str, float]]:
    return [per_query(q, r) for q, r in zip(queries, rankings, strict=True)]


def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0) -> float:
    """Two-sided p-value that mean(a) - mean(b) differs from 0, by resampling queries."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(min(p, 1.0))
