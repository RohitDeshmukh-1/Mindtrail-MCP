"""Storage contract. SQLite implements it now; PostgreSQL + pgvector will implement it next."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from mindtrail.core.models import MemoryRecord, MemoryType


@dataclass(frozen=True)
class ScopeFilter:
    """Which memories a query may see. Always tenant-bound; built by the service, never a client."""

    tenant_id: str
    space_ids: Sequence[str]
    now: datetime
    include_inactive: bool = False
    memory_types: Sequence[MemoryType] | None = None


class MemoryRepository(Protocol):
    def add(
        self, record: MemoryRecord, embedding: bytes | None, embedding_model: str | None
    ) -> None: ...

    def update(self, record: MemoryRecord, *, snapshot: MemoryRecord | None = None) -> None:
        """Persist new field values. ``snapshot`` (the previous state) is kept as a version."""
        ...

    def set_embedding(
        self, tenant_id: str, memory_id: UUID, embedding: bytes, model: str
    ) -> None: ...

    def get(self, tenant_id: str, memory_id: UUID) -> MemoryRecord | None: ...

    def get_many(self, tenant_id: str, memory_ids: Sequence[str]) -> dict[str, MemoryRecord]: ...

    def find_active_duplicate(
        self, tenant_id: str, space_id: str, content: str, now: datetime
    ) -> MemoryRecord | None: ...

    def delete(self, tenant_id: str, memory_id: UUID) -> bool: ...

    def keyword_search(
        self, scope: ScopeFilter, match_expression: str, limit: int
    ) -> list[tuple[str, float]]:
        """(memory_id, bm25) pairs, best first."""
        ...

    def embeddings(self, scope: ScopeFilter, model: str) -> list[tuple[str, bytes]]: ...

    def missing_embeddings(self, tenant_id: str, model: str, limit: int) -> list[tuple[str, str]]:
        """(memory_id, content) for memories without a vector from ``model``."""
        ...

    def list_records(
        self, tenant_id: str, space_id: str | None, limit: int, offset: int
    ) -> list[MemoryRecord]: ...

    def versions(self, tenant_id: str, memory_id: UUID) -> list[dict[str, Any]]: ...

    def stats(self, tenant_id: str, now: datetime) -> dict[str, Any]: ...

    def close(self) -> None: ...
