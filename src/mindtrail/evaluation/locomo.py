"""Retrieval evaluation on LoCoMo, a public long-term conversational memory benchmark.

Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents", ACL 2024.
https://github.com/snap-research/locomo. The dataset is licensed CC BY-NC 4.0, so it is not
bundled with Mindtrail. It is downloaded on first use and verified against a pinned checksum.

Each dialogue turn is stored as one memory in its conversation's space. For every question in
categories 1-4 we check whether the annotated evidence turns come back in the top k.
Category 5 (adversarial, "not mentioned") is excluded, as is common for retrieval evaluation.

These are *retrieval* metrics. Systems that publish LoCoMo results usually report LLM-judged
answer accuracy, which is a different measurement and not directly comparable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mindtrail.embeddings import EmbeddingProvider
from mindtrail.embeddings.rerankers import Reranker
from mindtrail.evaluation.metrics import bootstrap_ci, percentile
from mindtrail.memory.retrieval import RankingWeights
from mindtrail.memory.service import MemoryService
from mindtrail.storage.sqlite import SQLiteMemoryRepository

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
_DIA_ID = re.compile(r"D\d+:\d+")
_FIXED_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def fetch_locomo(cache_dir: Path) -> Path:
    """Path to a verified local copy of locomo10.json, downloading it if needed."""
    path = cache_dir / "locomo10.json"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LOCOMO_URL, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LOCOMO_SHA256:
        raise RuntimeError(
            f"{path} does not match the pinned LoCoMo checksum; delete it to re-download"
        )
    return path


class LocomoQuery(BaseModel):
    conversation: str
    category: str
    question: str
    evidence: list[str]
    retrieved: list[list[str]]  # dialogue ids for each returned memory, best first
    latency_ms: float

    def recall(self, k: int) -> float:
        found = {dia for turn in self.retrieved[:k] for dia in turn}
        return len(found & set(self.evidence)) / len(self.evidence)

    def hit(self, k: int) -> float:
        return 1.0 if self.recall(k) > 0 else 0.0


class LocomoMetrics(BaseModel):
    queries: int
    recall_at_5: float
    recall_at_10: float
    hit_at_1: float
    hit_at_5: float
    ci: dict[str, tuple[float, float]]
    latency_p50_ms: float
    latency_p95_ms: float


class LocomoReport(BaseModel):
    split: str
    embedder: str
    reranker: str | None
    turns: int
    skipped_evidence: int
    overall: LocomoMetrics
    by_category: dict[str, LocomoMetrics]
    results: list[LocomoQuery]

    def to_markdown(self) -> str:
        head = (
            f"**LoCoMo ({self.split} split)**: {self.turns} turns, "
            f"{self.overall.queries} questions, "
            f"embedder `{self.embedder}`, reranker `{self.reranker or 'none'}`\n\n"
            "| category | questions | recall@5 | recall@10 | hit@1 | hit@5 | p50 ms |\n"
            "|---|---:|---:|---:|---:|---:|---:|"
        )
        rows = [*sorted(self.by_category.items()), ("**overall**", self.overall)]
        lines = [head]
        for name, m in rows:
            lines.append(
                f"| {name} | {m.queries} | {m.recall_at_5:.1%} | {m.recall_at_10:.1%} "
                f"| {m.hit_at_1:.1%} | {m.hit_at_5:.1%} | {m.latency_p50_ms:.1f} |"
            )
        ci = self.overall.ci
        lines.append(
            "\n95% bootstrap CI (overall): "
            + ", ".join(f"{k} {lo:.1%}-{hi:.1%}" for k, (lo, hi) in ci.items())
        )
        return "\n".join(lines)


def _turn_text(turn: dict[str, Any], date: str) -> str:
    text = f"[{date}] {turn['speaker']}: {turn['text']}"
    if caption := turn.get("blip_caption"):
        text += f" (shares a photo: {caption})"
    return text


def _aggregate(results: Sequence[LocomoQuery]) -> LocomoMetrics:
    samples = {
        "recall@5": [r.recall(5) for r in results],
        "recall@10": [r.recall(10) for r in results],
        "hit@1": [r.hit(1) for r in results],
        "hit@5": [r.hit(5) for r in results],
    }
    latencies = [r.latency_ms for r in results]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return LocomoMetrics(
        queries=len(results),
        recall_at_5=mean(samples["recall@5"]),
        recall_at_10=mean(samples["recall@10"]),
        hit_at_1=mean(samples["hit@1"]),
        hit_at_5=mean(samples["hit@5"]),
        ci={name: bootstrap_ci(values) for name, values in samples.items()},
        latency_p50_ms=round(percentile(latencies, 50), 2),
        latency_p95_ms=round(percentile(latencies, 95), 2),
    )


def run_locomo(
    path: Path,
    embedder: EmbeddingProvider,
    *,
    reranker: Reranker | None = None,
    weights: RankingWeights | None = None,
    split: str = "all",
) -> LocomoReport:
    """Evaluate on a split: ``all`` (10 conversations), ``dev`` (the first 3, used for tuning
    Mindtrail's defaults) or ``test`` (the other 7, never used for tuning)."""
    samples = json.loads(path.read_text(encoding="utf-8"))
    samples = {"all": samples, "dev": samples[:3], "test": samples[3:]}[split]
    service = MemoryService(
        SQLiteMemoryRepository(":memory:"),
        embedder,
        weights=weights,
        reranker=reranker,
        clock=lambda: _FIXED_TIME,
    )
    results: list[LocomoQuery] = []
    turns = skipped = 0
    try:
        for sample in samples:
            conversation = sample["conversation"]
            space = "locomo:" + str(sample["sample_id"]).lower()
            items: list[dict[str, Any]] = []
            dia_ids: list[str] = []
            for key, value in conversation.items():
                if not (key.startswith("session_") and isinstance(value, list)):
                    continue
                date = conversation.get(f"{key}_date_time", "")
                for turn in value:
                    items.append(
                        {
                            "content": _turn_text(turn, date),
                            "space_id": space,
                            "memory_type": "episodic",
                        }
                    )
                    dia_ids.append(turn["dia_id"])
            stored = service.remember_many(items)
            turns += len(items)
            memory_to_dia: dict[str, list[str]] = defaultdict(list)
            for stored_result, dia in zip(stored, dia_ids, strict=True):
                memory_to_dia[str(stored_result.memory.id)].append(dia)
            known = set(dia_ids)

            for qa in sample["qa"]:
                category = CATEGORIES.get(qa.get("category", 0))
                evidence_raw = [
                    e for field in qa.get("evidence", []) for e in _DIA_ID.findall(field)
                ]
                evidence = sorted({e for e in evidence_raw if e in known})
                skipped += len(set(evidence_raw)) - len(evidence)
                if category is None or not evidence:
                    continue
                start = time.perf_counter()
                hits = service.search(qa["question"], space_ids=[space], limit=10)
                latency = (time.perf_counter() - start) * 1000
                results.append(
                    LocomoQuery(
                        conversation=space,
                        category=category,
                        question=qa["question"],
                        evidence=evidence,
                        retrieved=[memory_to_dia[str(h.memory.id)] for h in hits],
                        latency_ms=latency,
                    )
                )
    finally:
        service.close()

    by_category: dict[str, list[LocomoQuery]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
    return LocomoReport(
        split=split,
        embedder=embedder.model_name,
        reranker=reranker.model_name if reranker else None,
        turns=turns,
        skipped_evidence=skipped,
        overall=_aggregate(results),
        by_category={name: _aggregate(items) for name, items in by_category.items()},
        results=results,
    )
