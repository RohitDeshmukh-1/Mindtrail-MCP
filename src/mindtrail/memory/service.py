"""MemoryService: the single home of memory business logic.

MCP tools, the REST API, the SDK and the CLI are thin adapters over this class. The tenant is
fixed when the service is constructed (from the server-side auth context), never taken from
tool arguments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import ValidationError

from mindtrail.core.config import MindtrailConfig
from mindtrail.core.exceptions import InvalidMemoryError, MemoryNotFoundError, SecretDetectedError
from mindtrail.core.models import (
    DEFAULT_SPACE,
    MemoryContext,
    MemoryRecord,
    MemoryType,
    RememberResult,
    SearchHit,
    utcnow,
    validate_space_id,
)
from mindtrail.core.text import build_fts_query, content_hash
from mindtrail.embeddings import EmbeddingProvider, create_embedder
from mindtrail.embeddings.rerankers import Reranker, create_reranker, sigmoid
from mindtrail.memory.context import build_context
from mindtrail.memory.retrieval import RankingWeights, rank
from mindtrail.memory.safety import find_secrets
from mindtrail.storage.interfaces import MemoryRepository, ScopeFilter
from mindtrail.storage.sqlite import SQLiteMemoryRepository

logger = logging.getLogger(__name__)

MAX_SEARCH_LIMIT = 50
MAX_TOKEN_BUDGET = 32_000


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or 'input'}: {err['msg']}"
        for err in exc.errors()
    )


class MemoryService:
    def __init__(
        self,
        repository: MemoryRepository,
        embedder: EmbeddingProvider,
        *,
        tenant_id: str = "local",
        default_space: str = DEFAULT_SPACE,
        weights: RankingWeights | None = None,
        clock: Callable[[], datetime] = utcnow,
        candidate_limit: int = 50,
        reranker: Reranker | None = None,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._reranker = reranker
        self._tenant = tenant_id
        self._default_space = validate_space_id(default_space)
        self._weights = weights or RankingWeights()
        self._clock = clock
        self._candidate_limit = candidate_limit

    @classmethod
    def from_config(cls, config: MindtrailConfig | None = None) -> MemoryService:
        config = config or MindtrailConfig.from_env()
        return cls(
            SQLiteMemoryRepository(config.db_path),
            create_embedder(
                config.embedder, model=config.embedding_model, cache_dir=config.model_dir
            ),
            default_space=config.default_space,
            reranker=create_reranker(config.reranker, cache_dir=config.model_dir),
        )

    @property
    def default_space(self) -> str:
        return self._default_space

    @property
    def embedding_model(self) -> str:
        return self._embedder.model_name

    def now(self) -> datetime:
        return self._clock()

    @property
    def reranker_model(self) -> str | None:
        return self._reranker.model_name if self._reranker else None

    def warm_up(self) -> None:
        """Load the embedding and reranking models now rather than on the first query."""
        for component in (self._embedder, self._reranker):
            warm = getattr(component, "warm_up", None)
            if callable(warm):
                warm()

    def close(self) -> None:
        self._repo.close()

    # -- writes ----------------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
        space_id: str | None = None,
        source: str | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        supersedes: UUID | str | None = None,
    ) -> RememberResult:
        """Store a memory. Exact duplicates in the same space are merged, not stored twice."""
        now = self._clock()
        record = self._build(
            tenant_id=self._tenant,
            space_id=space_id or self._default_space,
            content=content,
            memory_type=memory_type,
            source=source,
            created_at=now,
            updated_at=now,
            valid_from=valid_from,
            valid_until=valid_until,
            importance=importance,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._reject_secrets(record)
        replaced = self.get(supersedes) if supersedes is not None else None

        existing = self._repo.find_active_duplicate(
            self._tenant, record.space_id, record.content, now
        )
        if existing is not None and (replaced is None or existing.id != replaced.id):
            memory = existing.model_copy(
                update={
                    "updated_at": now,
                    "importance": max(existing.importance, record.importance),
                    "confidence": max(existing.confidence, record.confidence),
                }
            )
            self._repo.update(memory)
            deduplicated = True
        else:
            embedding, model = self._embed_document(record.content)
            self._repo.add(record, embedding, model)
            memory, deduplicated = record, False

        if replaced is not None and replaced.id != memory.id:
            self._repo.update(
                replaced.model_copy(update={"superseded_by": memory.id, "updated_at": now}),
                snapshot=replaced,
            )
        return RememberResult(
            memory=memory,
            deduplicated=deduplicated,
            superseded_id=replaced.id if replaced is not None else None,
        )

    def remember_many(
        self, items: Sequence[dict[str, Any]], *, batch_size: int = 64
    ) -> list[RememberResult]:
        """Store many memories with batched embedding (for imports).

        Each item takes the keyword arguments of ``remember`` except ``supersedes``. Validation,
        the secret filter and duplicate merging apply exactly as for ``remember``; the first
        invalid item raises before anything is stored.
        """
        now = self._clock()
        records: list[MemoryRecord] = []
        for item in items:
            if "supersedes" in item:
                raise InvalidMemoryError("remember_many does not support 'supersedes'")
            fields = {**item, "memory_type": item.get("memory_type", MemoryType.SEMANTIC)}
            fields["space_id"] = item.get("space_id") or self._default_space
            record = self._build(tenant_id=self._tenant, created_at=now, updated_at=now, **fields)
            self._reject_secrets(record)
            records.append(record)

        results: list[RememberResult | None] = [None] * len(records)
        fresh: dict[tuple[str, str], int] = {}  # (space, content hash) -> index of first copy
        pending: list[int] = []
        for index, record in enumerate(records):
            key = (record.space_id, content_hash(record.content))
            if key in fresh:
                first = records[fresh[key]]
                results[index] = RememberResult(memory=first, deduplicated=True)
                continue
            existing = self._repo.find_active_duplicate(
                self._tenant, record.space_id, record.content, now
            )
            if existing is not None:
                merged = existing.model_copy(update={"updated_at": now})
                self._repo.update(merged)
                results[index] = RememberResult(memory=merged, deduplicated=True)
                continue
            fresh[key] = index
            pending.append(index)

        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            vectors = self._embed_many([records[i].content for i in chunk])
            for i, vector in zip(chunk, vectors, strict=True):
                blob = None if vector is None else np.asarray(vector, np.float32).tobytes()
                self._repo.add(records[i], blob, self._embedder.model_name if blob else None)
                results[i] = RememberResult(memory=records[i])
        return [r for r in results if r is not None]

    def update_memory(
        self,
        memory_id: UUID | str,
        *,
        content: str | None = None,
        memory_type: MemoryType | str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Correct a memory in place. The previous state is kept in its version history."""
        current = self.get(memory_id)
        changes: dict[str, Any] = {
            key: value
            for key, value in {
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "confidence": confidence,
                "metadata": metadata,
            }.items()
            if value is not None
        }
        if not changes:
            return current
        updated = self._build(**{**current.model_dump(), **changes, "updated_at": self._clock()})
        self._reject_secrets(updated)
        self._repo.update(updated, snapshot=current)
        if updated.content != current.content:
            embedding, model = self._embed_document(updated.content)
            if embedding is not None and model is not None:
                self._repo.set_embedding(self._tenant, updated.id, embedding, model)
        return updated

    def forget(self, memory_id: UUID | str, *, hard: bool = True) -> bool:
        """Delete a memory permanently (``hard``) or invalidate it while keeping history."""
        current = self.get(memory_id)
        if hard:
            return self._repo.delete(self._tenant, current.id)
        now = self._clock()
        changes: dict[str, Any] = {"valid_until": now, "updated_at": now}
        if current.valid_from is not None and current.valid_from >= now:
            changes["valid_from"] = now - timedelta(microseconds=1)
        self._repo.update(current.model_copy(update=changes), snapshot=current)
        return True

    def reindex_embeddings(self, batch_size: int = 64) -> int:
        """Embed memories that have no vector from the current model. Returns the count."""
        total = 0
        model = self._embedder.model_name
        while batch := self._repo.missing_embeddings(self._tenant, model, batch_size):
            vectors = self._embedder.embed_documents([content for _, content in batch])
            for (memory_id, _), vector in zip(batch, vectors, strict=True):
                blob = np.asarray(vector, dtype=np.float32).tobytes()
                self._repo.set_embedding(self._tenant, UUID(memory_id), blob, model)
            total += len(batch)
        return total

    # -- reads -----------------------------------------------------------------------------

    def get(self, memory_id: UUID | str) -> MemoryRecord:
        uid = self._parse_id(memory_id)
        record = self._repo.get(self._tenant, uid)
        if record is None:
            raise MemoryNotFoundError(uid)
        return record

    def history(self, memory_id: UUID | str) -> list[dict[str, Any]]:
        return self._repo.versions(self._tenant, self.get(memory_id).id)

    def list_memories(
        self, *, space_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[MemoryRecord]:
        space = self._space(space_id) if space_id is not None else None
        return self._repo.list_records(self._tenant, space, max(1, min(limit, 500)), max(0, offset))

    def stats(self) -> dict[str, Any]:
        return {
            **self._repo.stats(self._tenant, self._clock()),
            "embedding_model": self._embedder.model_name,
        }

    def search(
        self,
        query: str,
        *,
        space_ids: Sequence[str] | None = None,
        limit: int = 5,
        memory_types: Sequence[MemoryType | str] | None = None,
        include_inactive: bool = False,
    ) -> list[SearchHit]:
        """Hybrid keyword + vector search. Returns [] when nothing relevant exists."""
        query = query.strip()
        if not query:
            raise InvalidMemoryError("query must not be blank")
        try:
            types = [MemoryType(t) for t in memory_types] if memory_types else None
        except ValueError as exc:
            raise InvalidMemoryError(str(exc)) from exc
        scope = ScopeFilter(
            tenant_id=self._tenant,
            space_ids=[self._space(s) for s in space_ids] if space_ids else [self._default_space],
            now=self._clock(),
            include_inactive=include_inactive,
            memory_types=types,
        )

        limit = max(1, min(limit, MAX_SEARCH_LIMIT))
        similarities = self._similarities(query, scope)
        if self._reranker is not None:
            reranked = self._reranked_search(query, scope, similarities)
            if reranked is not None:
                return reranked[:limit]

        # Without a reranker, strict gates decide relevance directly.
        vector_floor = self._weights.vector_min_similarity
        if vector_floor is None:
            vector_floor = self._embedder.min_similarity
        keyword_ids = self._keyword_candidates(query, scope, similarities, support_gate=True)
        vector_hits = self._vector_candidates(similarities, vector_floor)
        records = self._repo.get_many(self._tenant, _union(keyword_ids, vector_hits))
        return rank(records, keyword_ids, vector_hits, self._weights, scope.now)[:limit]

    def _reranked_search(
        self, query: str, scope: ScopeFilter, similarities: dict[str, float]
    ) -> list[SearchHit] | None:
        """Loose hybrid candidates, rescored by the cross-encoder. None if the reranker fails."""
        assert self._reranker is not None
        floor = self._weights.rerank_candidate_similarity
        if floor is None:
            floor = self._embedder.candidate_similarity
        keyword_ids = self._keyword_candidates(query, scope, similarities, support_gate=False)
        vector_hits = self._vector_candidates(similarities, floor)
        records = self._repo.get_many(self._tenant, _union(keyword_ids, vector_hits))
        pool = rank(records, keyword_ids, vector_hits, self._weights, scope.now)
        pool = pool[: self._weights.rerank_depth]
        if not pool:
            return []
        try:
            scores = self._reranker.score(query, [hit.memory.content for hit in pool])
        except Exception:
            logger.exception("reranker failed; falling back to hybrid ranking")
            return None
        min_score = self._weights.rerank_min_score
        if min_score is None:
            min_score = self._reranker.min_score
        relevance = {
            str(hit.memory.id): sigmoid(score)
            for hit, score in zip(pool, scores, strict=True)
            if score >= min_score
        }
        return rank(records, keyword_ids, vector_hits, self._weights, scope.now, relevance)

    def recall(self, query: str, *, space_id: str | None = None, limit: int = 5) -> list[SearchHit]:
        return self.search(query, space_ids=[space_id] if space_id else None, limit=limit)

    def get_context(
        self,
        query: str,
        *,
        space_ids: Sequence[str] | None = None,
        token_budget: int = 1_500,
        limit: int = 20,
    ) -> MemoryContext:
        """Relevant memories formatted for prompt injection, within ``token_budget`` tokens."""
        hits = self.search(query, space_ids=space_ids, limit=limit)
        return build_context(query, hits, max(0, min(token_budget, MAX_TOKEN_BUDGET)))

    # -- helpers ---------------------------------------------------------------------------

    def _build(self, **fields: Any) -> MemoryRecord:
        try:
            return MemoryRecord(**fields)
        except ValidationError as exc:
            raise InvalidMemoryError(_validation_message(exc)) from exc

    def _space(self, space_id: str) -> str:
        try:
            return validate_space_id(space_id)
        except ValueError as exc:
            raise InvalidMemoryError(str(exc)) from exc

    @staticmethod
    def _parse_id(memory_id: UUID | str) -> UUID:
        if isinstance(memory_id, UUID):
            return memory_id
        try:
            return UUID(memory_id)
        except ValueError as exc:
            raise InvalidMemoryError(f"invalid memory id: {memory_id!r}") from exc

    @staticmethod
    def _reject_secrets(record: MemoryRecord) -> None:
        kinds = find_secrets(record.content) + find_secrets(str(record.metadata))
        if kinds:
            raise SecretDetectedError(sorted(set(kinds)))

    def _embed_document(self, content: str) -> tuple[bytes | None, str | None]:
        # A failed embedding must never lose the write: store without a vector and let
        # reindex_embeddings() fill it in later. Keyword search still finds the memory.
        try:
            vector = self._embedder.embed_documents([content])[0]
        except Exception:
            logger.exception("embedding failed; memory stored without a vector")
            return None, None
        return np.asarray(vector, dtype=np.float32).tobytes(), self._embedder.model_name

    def _vector_candidates(
        self, similarities: dict[str, float], floor: float
    ) -> list[tuple[str, float]]:
        hits = [(mid, sim) for mid, sim in similarities.items() if sim >= floor]
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[: self._candidate_limit]

    def _keyword_candidates(
        self,
        query: str,
        scope: ScopeFilter,
        similarities: dict[str, float],
        *,
        support_gate: bool,
    ) -> list[str]:
        match = build_fts_query(query)
        if match is None:
            return []
        keyword = self._repo.keyword_search(scope, match, self._candidate_limit)
        if keyword and self._weights.keyword_relative_floor > 0:
            # bm25() is negative (more negative = better), so the ratio to the best is in (0, 1].
            best = keyword[0][1]
            floor = self._weights.keyword_relative_floor
            keyword = [(mid, score) for mid, score in keyword if score / best >= floor]
        if not support_gate:
            return [mid for mid, _ in keyword]
        support = self._weights.keyword_min_similarity
        if support is None:
            support = self._embedder.keyword_support_similarity
        # A keyword hit whose meaning is unrelated to the query (low vector similarity) is an
        # incidental word match. Memories without a vector keep their keyword hits.
        return [
            mid
            for mid, _ in keyword
            if not similarities or mid not in similarities or similarities[mid] >= support
        ]

    def _embed_many(self, contents: list[str]) -> list[Any]:
        """Batch embedding; on failure every item is stored without a vector (see above)."""
        try:
            return list(self._embedder.embed_documents(contents))
        except Exception:
            logger.exception("batch embedding failed; memories stored without vectors")
            return [None] * len(contents)

    def _similarities(self, query: str, scope: ScopeFilter) -> dict[str, float]:
        """Cosine similarity of the query to every in-scope memory with a current vector."""
        rows = self._repo.embeddings(scope, self._embedder.model_name)
        if not rows:
            return {}
        try:
            query_vector = self._embedder.embed_query(query)
        except Exception:
            logger.exception("query embedding failed; falling back to keyword search")
            return {}
        matrix = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        scores = matrix @ query_vector
        return {row[0]: float(score) for row, score in zip(rows, scores, strict=True)}


def _union(keyword_ids: Sequence[str], vector_hits: Sequence[tuple[str, float]]) -> list[str]:
    return list(dict.fromkeys([*keyword_ids, *(mid for mid, _ in vector_hits)]))
