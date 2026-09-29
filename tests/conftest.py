from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from cogmem.embeddings import HashingEmbedder
from cogmem.memory.service import MemoryService
from cogmem.storage.sqlite import SQLiteMemoryRepository


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


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
