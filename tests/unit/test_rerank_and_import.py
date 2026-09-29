from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from tests.conftest import FakeClock

from mindtrail.core.exceptions import InvalidMemoryError, SecretDetectedError
from mindtrail.embeddings import HashingEmbedder
from mindtrail.evaluation.locomo import run_locomo
from mindtrail.evaluation.metrics import bootstrap_ci
from mindtrail.memory.service import MemoryService
from mindtrail.storage.sqlite import SQLiteMemoryRepository


class KeywordReranker:
    """Deterministic stand-in for a cross-encoder: scores by a marker word in the document."""

    model_name = "fake-reranker"
    min_score = 0.0

    def __init__(self, favored: str) -> None:
        self.favored = favored
        self.calls = 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls += 1
        return [5.0 if self.favored in doc else -5.0 for doc in documents]


class BrokenReranker(KeywordReranker):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        raise RuntimeError("model crashed")


def _service(repo: SQLiteMemoryRepository, clock: FakeClock, reranker: object) -> MemoryService:
    return MemoryService(repo, HashingEmbedder(), clock=clock, reranker=reranker)  # type: ignore[arg-type]


def test_reranker_decides_order_and_relevance(
    repo: SQLiteMemoryRepository, clock: FakeClock
) -> None:
    reranker = KeywordReranker(favored="Alembic")
    service = _service(repo, clock, reranker)
    service.remember("Database schema changes are applied with Alembic")
    service.remember("The database is PostgreSQL")
    hits = service.search("how do database schema changes work")
    assert [h.memory.content for h in hits] == ["Database schema changes are applied with Alembic"]
    assert hits[0].signals["rerank"] > 0.99 and reranker.calls == 1


def test_reranker_can_abstain(repo: SQLiteMemoryRepository, clock: FakeClock) -> None:
    service = _service(repo, clock, KeywordReranker(favored="never-present"))
    service.remember("The database is PostgreSQL")
    assert service.search("which database") == []


def test_reranker_failure_falls_back_to_hybrid(
    repo: SQLiteMemoryRepository, clock: FakeClock
) -> None:
    service = _service(repo, clock, BrokenReranker(favored="x"))
    service.remember("The database is PostgreSQL")
    assert [h.memory.content for h in service.search("PostgreSQL database")] == [
        "The database is PostgreSQL"
    ]


def test_scope_is_enforced_before_reranking(repo: SQLiteMemoryRepository, clock: FakeClock) -> None:
    service = _service(repo, clock, KeywordReranker(favored="secret"))
    service.remember("Other project secret roadmap", space_id="project:other")
    assert service.search("secret roadmap", space_ids=["project:mine"]) == []


def test_remember_many_embeds_in_batches_and_dedups(service: MemoryService) -> None:
    results = service.remember_many(
        [
            {"content": "Fact one", "space_id": "repo:a"},
            {"content": "Fact two", "space_id": "repo:a", "importance": 0.9},
            {"content": "fact  ONE", "space_id": "repo:a"},  # duplicate inside the batch
            {"content": "Fact one", "space_id": "repo:b"},  # same text, different space
        ],
        batch_size=2,
    )
    assert [r.deduplicated for r in results] == [False, False, True, False]
    assert results[2].memory.id == results[0].memory.id
    assert service.stats()["total"] == 3
    assert service.search("fact two", space_ids=["repo:a"])[0].memory.importance == 0.9
    again = service.remember_many([{"content": "Fact two", "space_id": "repo:a"}])
    assert again[0].deduplicated and service.stats()["total"] == 3


def test_remember_many_validates_everything_before_storing(service: MemoryService) -> None:
    with pytest.raises(SecretDetectedError):
        service.remember_many([{"content": "ok"}, {"content": "key AKIAABCDEFGHIJKLMNOP"}])
    with pytest.raises(InvalidMemoryError):
        service.remember_many([{"content": "ok"}, {"content": "  "}])
    with pytest.raises(InvalidMemoryError):
        service.remember_many([{"content": "ok", "supersedes": "x"}])
    assert service.stats()["total"] == 0


def test_bootstrap_ci_is_reproducible_and_brackets_the_mean() -> None:
    values = [1.0] * 70 + [0.0] * 30
    low, high = bootstrap_ci(values)
    assert low < 0.7 < high and (low, high) == bootstrap_ci(values)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_locomo_runner_on_synthetic_conversation(tmp_path: Path) -> None:
    sample = {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Ann",
            "speaker_b": "Bo",
            "session_1_date_time": "1:00 pm on 3 May, 2023",
            "session_1": [
                {"speaker": "Ann", "dia_id": "D1:1", "text": "I adopted a beagle named Toby"},
                {"speaker": "Bo", "dia_id": "D1:2", "text": "I started learning the violin"},
            ],
        },
        "qa": [
            {"question": "What is the name of Ann's dog?", "answer": "Toby",
             "evidence": ["D1:1"], "category": 4},
            {"question": "Which instrument is Bo learning?", "answer": "violin",
             "evidence": ["D1:2; D9:9"], "category": 1},
            {"question": "Adversarial", "evidence": ["D1:1"], "category": 5},
        ],
    }  # fmt: skip
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([sample]), encoding="utf-8")
    report = run_locomo(path, HashingEmbedder())
    assert report.turns == 2 and report.overall.queries == 2  # category 5 excluded
    assert report.skipped_evidence == 1  # D9:9 does not exist
    assert report.overall.recall_at_10 == 1.0
    assert set(report.by_category) == {"single-hop", "multi-hop"}
