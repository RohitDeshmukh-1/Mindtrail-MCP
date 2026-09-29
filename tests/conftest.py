from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from mindtrail.embeddings import HashingEmbedder
from mindtrail.memory.service import MemoryService
from mindtrail.storage.sqlite import SQLiteMemoryRepository


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture(autouse=True)
def _offline_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: never download or load neural models unless a test opts in."""
    monkeypatch.setenv("MINDTRAIL_EMBEDDER", "hashing")
    monkeypatch.setenv("MINDTRAIL_RERANKER", "none")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def repo() -> Iterator[SQLiteMemoryRepository]:
    repository = SQLiteMemoryRepository(":memory:")
    yield repository
    repository.close()


@pytest.fixture
def service(repo: SQLiteMemoryRepository, clock: FakeClock) -> MemoryService:
    return MemoryService(repo, HashingEmbedder(), clock=clock)
