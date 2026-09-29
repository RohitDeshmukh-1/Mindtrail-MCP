"""Seed a fresh in-memory store with a dataset, run every query, and score the results."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import BaseModel

from mindtrail.embeddings import EmbeddingProvider
from mindtrail.embeddings.rerankers import Reranker
from mindtrail.evaluation.dataset import BenchDataset, BenchQuery
from mindtrail.evaluation.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from mindtrail.memory.retrieval import RankingWeights
from mindtrail.memory.service import MemoryService
from mindtrail.storage.sqlite import SQLiteMemoryRepository

REFERENCE_TIME = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


class QueryResult(BaseModel):
    id: str
    category: str
    query: str
    retrieved: list[str]
    relevant: list[str]
    forbidden: list[str]
    latency_ms: float

    @property
    def violated(self) -> bool:
        return bool(set(self.retrieved) & set(self.forbidden))

    @property
    def correct(self) -> bool:
        """Top result answers the query (or nothing is returned when nothing should be), and
        no stale or foreign memory appears anywhere in the results."""
        if self.violated:
            return False
        if self.relevant:
            return bool(self.retrieved) and self.retrieved[0] in self.relevant
        return not self.retrieved


class Metrics(BaseModel):
    queries: int
    accuracy: float = 0.0  # share of all queries handled correctly (see QueryResult.correct)
    recall_at_1: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    abstention: float | None = None  # share of no-answer queries that correctly returned nothing
    violations: float | None = None  # share of queries that returned a stale or foreign memory
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    # 95% bootstrap confidence intervals, keyed by metric name.
    ci: dict[str, tuple[float, float]] = {}


class BenchmarkReport(BaseModel):
    dataset: str
    embedder: str
    reranker: str | None = None
    k: int
    overall: Metrics
    by_category: dict[str, Metrics]
    results: list[QueryResult]

    def failures(self) -> list[QueryResult]:
        """Queries that missed their target at rank 1, answered a no-answer query, or leaked."""
        return [
            r
            for r in self.results
            if r.violated
            or (r.relevant and (not r.retrieved or r.retrieved[0] not in r.relevant))
            or (not r.relevant and r.retrieved)
        ]

    def to_markdown(self) -> str:
        def cell(value: float | None, pct: bool = True) -> str:
            if value is None:
                return "-"
            return f"{value * 100:.0f}%" if pct else f"{value:.3f}"

        k = self.k
        lines = [
            f"**{self.dataset}**, embedder `{self.embedder}`, "
            f"reranker `{self.reranker or 'none'}`, k={k}\n",
            f"| category | queries | accuracy | recall@1 | recall@{k} | MRR | nDCG@{k} "
            "| abstention | stale/foreign leaks | p50 ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        rows = [*sorted(self.by_category.items()), ("**overall**", self.overall)]
        for name, m in rows:
            lines.append(
                f"| {name} | {m.queries} | {cell(m.accuracy)} | {cell(m.recall_at_1)} "
                f"| {cell(m.recall_at_k)} "
                f"| {cell(m.mrr, pct=False)} | {cell(m.ndcg_at_k, pct=False)} "
                f"| {cell(m.abstention)} | {cell(m.violations)} "
                f"| {m.latency_p50_ms:.1f} | {m.latency_p95_ms:.1f} |"
            )
        labels = {"accuracy": "accuracy", "recall_at_1": "recall@1", "recall_at_k": f"recall@{k}"}
        labels["abstention"] = "abstention"
        intervals = ", ".join(
            f"{labels[name]} {low * 100:.0f}-{high * 100:.0f}%"
            for name, (low, high) in self.overall.ci.items()
        )
        lines.append(f"\n95% bootstrap confidence intervals (overall): {intervals}")
        return "\n".join(lines)


class _Clock:
    def __init__(self) -> None:
        self.now = REFERENCE_TIME

    def __call__(self) -> datetime:
        return self.now


def _seed(service: MemoryService, clock: _Clock, dataset: BenchDataset) -> dict[str, str]:
    """Store every memory at its simulated age. Returns memory id -> dataset key."""
    ids: dict[str, UUID] = {}
    for memory in sorted(dataset.memories, key=lambda m: -m.age_days):
        clock.now = REFERENCE_TIME - timedelta(days=memory.age_days)
        valid_until = (
            clock.now + timedelta(days=memory.valid_for_days) if memory.valid_for_days else None
        )
        result = service.remember(
            memory.content,
            memory_type=memory.type,
            space_id=memory.space,
            importance=memory.importance,
            valid_until=valid_until,
            supersedes=ids[memory.supersedes] if memory.supersedes else None,
        )
        ids[memory.key] = result.memory.id
    clock.now = REFERENCE_TIME
    return {str(memory_id): key for key, memory_id in ids.items()}


def _aggregate(results: Sequence[QueryResult], k: int) -> Metrics:
    ranked = [r for r in results if r.relevant]
    no_answer = [r for r in results if not r.relevant]
    guarded = [r for r in results if r.forbidden]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    latencies = [r.latency_ms for r in results]
    samples = {
        "accuracy": [1.0 if r.correct else 0.0 for r in results],
        "recall_at_1": [recall_at_k(r.retrieved, r.relevant, 1) for r in ranked],
        "recall_at_k": [recall_at_k(r.retrieved, r.relevant, k) for r in ranked],
        "abstention": [0.0 if r.retrieved else 1.0 for r in no_answer],
    }
    return Metrics(
        ci={name: bootstrap_ci(values) for name, values in samples.items() if values},
        queries=len(results),
        accuracy=sum(r.correct for r in results) / len(results) if results else 0.0,
        recall_at_1=mean([recall_at_k(r.retrieved, r.relevant, 1) for r in ranked]),
        recall_at_k=mean([recall_at_k(r.retrieved, r.relevant, k) for r in ranked]),
        mrr=mean([reciprocal_rank(r.retrieved, r.relevant) for r in ranked]),
        ndcg_at_k=mean([ndcg_at_k(r.retrieved, r.relevant, k) for r in ranked]),
        abstention=mean([0.0 if r.retrieved else 1.0 for r in no_answer]),
        violations=mean([1.0 if r.violated else 0.0 for r in guarded]),
        latency_p50_ms=round(percentile(latencies, 50), 2),
        latency_p95_ms=round(percentile(latencies, 95), 2),
    )


def run_benchmark(
    dataset: BenchDataset,
    embedder: EmbeddingProvider,
    *,
    k: int = 5,
    weights: RankingWeights | None = None,
    reranker: Reranker | None = None,
) -> BenchmarkReport:
    clock = _Clock()
    repository = SQLiteMemoryRepository(":memory:")
    service = MemoryService(repository, embedder, clock=clock, weights=weights, reranker=reranker)
    try:
        keys = _seed(service, clock, dataset)
        results = [_run_query(service, keys, query, k) for query in dataset.queries]
    finally:
        service.close()

    by_category: dict[str, list[QueryResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
    return BenchmarkReport(
        dataset=f"{dataset.name} v{dataset.version}",
        embedder=embedder.model_name,
        reranker=reranker.model_name if reranker else None,
        k=k,
        overall=_aggregate(results, k),
        by_category={name: _aggregate(items, k) for name, items in by_category.items()},
        results=results,
    )


def _run_query(
    service: MemoryService, keys: dict[str, str], query: BenchQuery, k: int
) -> QueryResult:
    start = time.perf_counter()
    hits = service.search(query.query, space_ids=query.spaces, limit=k)
    latency_ms = (time.perf_counter() - start) * 1000
    return QueryResult(
        id=query.id,
        category=query.category,
        query=query.query,
        retrieved=[keys[str(hit.memory.id)] for hit in hits],
        relevant=query.relevant,
        forbidden=query.forbidden,
        latency_ms=latency_ms,
    )
