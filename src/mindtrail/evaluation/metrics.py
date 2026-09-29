"""Standard ranking metrics over lists of retrieved ids (binary relevance)."""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-query ``values``.

    Seeded, so reported intervals are reproducible.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(resamples - 1, int((1 - tail) * resamples))]
    return (low, high)
