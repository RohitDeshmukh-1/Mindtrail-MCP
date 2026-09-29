"""Transparent hybrid ranking: reciprocal-rank fusion of keyword and vector results (or, when
a cross-encoder reranker is configured, its relevance scores), then small multiplicative boosts
for importance, confidence and recency.

Boosts only reorder candidates that already matched the query, so an important but irrelevant
memory can never outrank a relevant one by boosts alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from mindtrail.core.models import MemoryRecord, SearchHit


@dataclass(frozen=True)
class RankingWeights:
    keyword: float = 1.0
    vector: float = 1.0
    importance: float = 0.3
    recency: float = 0.2
    recency_half_life_days: float = 30.0
    rrf_k: int = 60
    # Candidate gates (None = the embedder's calibrated default):
    # - keyword hits below this fraction of the best BM25 score are dropped;
    # - keyword hits whose vector similarity is below keyword_min_similarity are dropped;
    # - vector hits must reach vector_min_similarity.
    keyword_relative_floor: float = 0.4
    keyword_min_similarity: float | None = None
    vector_min_similarity: float | None = None
    # Reranking (used only when a reranker is configured): how many fused candidates are
    # rescored, the loose similarity floor for vector candidates, and the reranker score floor.
    rerank_depth: int = 20
    rerank_candidate_similarity: float | None = None
    rerank_min_score: float | None = None


def rank(
    records: Mapping[str, MemoryRecord],
    keyword_ids: Sequence[str],
    vector_hits: Sequence[tuple[str, float]],
    weights: RankingWeights,
    now: datetime,
    relevance: Mapping[str, float] | None = None,
) -> list[SearchHit]:
    """Fuse keyword ids (best first) and (id, cosine) vector hits (best first) into ranked hits.

    ``relevance`` (memory id -> 0..1), when given, replaces the fused rank score and restricts
    the result to the memories it contains. The reranker uses this.
    """
    k = weights.rrf_k
    max_rrf = (weights.keyword + weights.vector) / (k + 1)
    signals: dict[str, dict[str, float]] = {}

    for position, memory_id in enumerate(keyword_ids, start=1):
        entry = signals.setdefault(memory_id, {"rrf": 0.0})
        entry["keyword_rank"] = position
        entry["rrf"] += weights.keyword / (k + position)
    for position, (memory_id, similarity) in enumerate(vector_hits, start=1):
        entry = signals.setdefault(memory_id, {"rrf": 0.0})
        entry["vector_rank"] = position
        entry["vector_similarity"] = round(similarity, 4)
        entry["rrf"] += weights.vector / (k + position)

    hits: list[SearchHit] = []
    for memory_id, entry in signals.items():
        record = records.get(memory_id)
        if record is None:  # deleted between candidate retrieval and fetch
            continue
        if relevance is not None and memory_id not in relevance:
            continue
        age_days = max((now - record.updated_at).total_seconds(), 0.0) / 86_400
        recency = 0.5 ** (age_days / weights.recency_half_life_days)
        fused = entry["rrf"] / max_rrf
        if relevance is None:
            score_base = fused
        else:
            score_base = relevance[memory_id]
            entry["rerank"] = round(score_base, 4)
        boost = 1 + weights.importance * (record.importance - 0.5) + weights.recency * recency
        score = score_base * boost * (0.5 + 0.5 * record.confidence)
        entry.update(relevance=round(score_base, 4), recency=round(recency, 4))
        hits.append(SearchHit(memory=record, score=round(score, 6), signals=entry))

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits
