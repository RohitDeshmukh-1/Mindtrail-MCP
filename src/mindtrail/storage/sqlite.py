"""SQLite backend for local / self-hosted mode: FTS5 for keywords, float32 BLOBs for vectors."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from mindtrail.core.models import MemoryRecord
from mindtrail.core.text import content_hash
from mindtrail.storage.interfaces import ScopeFilter

# Fixed-width UTC timestamps so SQL string comparison matches chronological order.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+00:00"

_MIGRATIONS: list[str] = [
    """
    CREATE TABLE memories (
        seq            INTEGER PRIMARY KEY AUTOINCREMENT,  -- stable rowid for the FTS index
        id             TEXT NOT NULL UNIQUE,
        tenant_id      TEXT NOT NULL,
        space_id       TEXT NOT NULL,
        content        TEXT NOT NULL,
        content_hash   TEXT NOT NULL,
        memory_type    TEXT NOT NULL,
        source         TEXT,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL,
        valid_from     TEXT,
        valid_until    TEXT,
        importance     REAL NOT NULL CHECK (importance BETWEEN 0 AND 1),
        confidence     REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
        superseded_by  TEXT REFERENCES memories(id) ON DELETE SET NULL,
        metadata       TEXT NOT NULL DEFAULT '{}',
        embedding      BLOB,
        embedding_model TEXT
    );
    CREATE INDEX ix_memories_scope ON memories (tenant_id, space_id, updated_at);
    CREATE INDEX ix_memories_hash ON memories (tenant_id, space_id, content_hash);

    CREATE VIRTUAL TABLE memories_fts USING fts5(
        content, content='memories', content_rowid='seq', tokenize='porter unicode61'
    );
    CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts (rowid, content) VALUES (new.seq, new.content);
    END;
    CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts (memories_fts, rowid, content)
        VALUES ('delete', old.seq, old.content);
    END;
    CREATE TRIGGER memories_au AFTER UPDATE OF content ON memories BEGIN
        INSERT INTO memories_fts (memories_fts, rowid, content)
        VALUES ('delete', old.seq, old.content);
        INSERT INTO memories_fts (rowid, content) VALUES (new.seq, new.content);
    END;

    CREATE TABLE memory_versions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        content     TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        importance  REAL NOT NULL,
        confidence  REAL NOT NULL,
        metadata    TEXT NOT NULL,
        valid_until TEXT,
        recorded_at TEXT NOT NULL
    );
    CREATE INDEX ix_versions_memory ON memory_versions (memory_id, id);
    """,
]


def _ts(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).strftime(_TS_FORMAT)


def _parse_ts(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


class SQLiteMemoryRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        # Overwrite deleted content on disk so "forget" is a real deletion.
        self._conn.execute("PRAGMA secure_delete = ON")
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    # -- lifecycle -------------------------------------------------------------------------

    def _migrate(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            for number, script in enumerate(_MIGRATIONS, start=1):
                if version < number:
                    self._conn.executescript(
                        f"BEGIN IMMEDIATE;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;"
                    )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes ----------------------------------------------------------------------------

    def add(
        self, record: MemoryRecord, embedding: bytes | None, embedding_model: str | None
    ) -> None:
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO memories (id, tenant_id, space_id, content, content_hash,
                       memory_type, source, created_at, updated_at, valid_from, valid_until,
                       importance, confidence, superseded_by, metadata, embedding, embedding_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(record.id), record.tenant_id, record.space_id, record.content,
                    content_hash(record.content), record.memory_type.value, record.source,
                    _ts(record.created_at), _ts(record.updated_at), _ts(record.valid_from),
                    _ts(record.valid_until), record.importance, record.confidence,
                    str(record.superseded_by) if record.superseded_by else None,
                    json.dumps(record.metadata), embedding, embedding_model,
                ),
            )  # fmt: skip

    def update(self, record: MemoryRecord, *, snapshot: MemoryRecord | None = None) -> None:
        with self._transaction() as conn:
            if snapshot is not None:
                conn.execute(
                    """INSERT INTO memory_versions (memory_id, content, memory_type, importance,
                           confidence, metadata, valid_until, recorded_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(snapshot.id), snapshot.content, snapshot.memory_type.value,
                        snapshot.importance, snapshot.confidence, json.dumps(snapshot.metadata),
                        _ts(snapshot.valid_until), _ts(record.updated_at),
                    ),
                )  # fmt: skip
            cursor = conn.execute(
                """UPDATE memories SET content = ?, content_hash = ?, memory_type = ?,
                       source = ?, updated_at = ?, valid_from = ?, valid_until = ?,
                       importance = ?, confidence = ?, superseded_by = ?, metadata = ?
                   WHERE tenant_id = ? AND id = ?""",
                (
                    record.content, content_hash(record.content), record.memory_type.value,
                    record.source, _ts(record.updated_at), _ts(record.valid_from),
                    _ts(record.valid_until), record.importance, record.confidence,
                    str(record.superseded_by) if record.superseded_by else None,
                    json.dumps(record.metadata), record.tenant_id, str(record.id),
                ),
            )  # fmt: skip
            if cursor.rowcount != 1:
                raise LookupError(f"memory {record.id} does not exist for tenant")

    def set_embedding(self, tenant_id: str, memory_id: UUID, embedding: bytes, model: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "UPDATE memories SET embedding = ?, embedding_model = ? "
                "WHERE tenant_id = ? AND id = ?",
                (embedding, model, tenant_id, str(memory_id)),
            )

    def delete(self, tenant_id: str, memory_id: UUID) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE tenant_id = ? AND id = ?", (tenant_id, str(memory_id))
            )
            return cursor.rowcount == 1

    # -- reads -----------------------------------------------------------------------------

    def get(self, tenant_id: str, memory_id: UUID) -> MemoryRecord | None:
        rows = self._query(
            "SELECT * FROM memories WHERE tenant_id = ? AND id = ?", (tenant_id, str(memory_id))
        )
        return _to_record(rows[0]) if rows else None

    def get_many(self, tenant_id: str, memory_ids: Sequence[str]) -> dict[str, MemoryRecord]:
        if not memory_ids:
            return {}
        marks = ",".join("?" * len(memory_ids))
        rows = self._query(
            f"SELECT * FROM memories WHERE tenant_id = ? AND id IN ({marks})",
            (tenant_id, *memory_ids),
        )
        return {row["id"]: _to_record(row) for row in rows}

    def find_active_duplicate(
        self, tenant_id: str, space_id: str, content: str, now: datetime
    ) -> MemoryRecord | None:
        where, params = _scope_sql(ScopeFilter(tenant_id, [space_id], now))
        rows = self._query(
            f"SELECT * FROM memories m WHERE {where} AND m.content_hash = ? LIMIT 1",
            (*params, content_hash(content)),
        )
        return _to_record(rows[0]) if rows else None

    def keyword_search(
        self, scope: ScopeFilter, match_expression: str, limit: int
    ) -> list[tuple[str, float]]:
        where, params = _scope_sql(scope)
        rows = self._query(
            f"""SELECT m.id, bm25(memories_fts) AS rank
                FROM memories_fts JOIN memories m ON m.seq = memories_fts.rowid
                WHERE memories_fts MATCH ? AND {where}
                ORDER BY rank LIMIT ?""",
            (match_expression, *params, limit),
        )
        return [(row["id"], float(row["rank"])) for row in rows]

    def embeddings(self, scope: ScopeFilter, model: str) -> list[tuple[str, bytes]]:
        # Brute-force scan is fine for local scale; sqlite-vec / pgvector replace it at scale.
        where, params = _scope_sql(scope)
        rows = self._query(
            f"SELECT m.id, m.embedding FROM memories m "
            f"WHERE {where} AND m.embedding_model = ? AND m.embedding IS NOT NULL",
            (*params, model),
        )
        return [(row["id"], row["embedding"]) for row in rows]

    def missing_embeddings(self, tenant_id: str, model: str, limit: int) -> list[tuple[str, str]]:
        rows = self._query(
            "SELECT id, content FROM memories WHERE tenant_id = ? "
            "AND (embedding IS NULL OR embedding_model IS NOT ?) LIMIT ?",
            (tenant_id, model, limit),
        )
        return [(row["id"], row["content"]) for row in rows]

    def list_records(
        self, tenant_id: str, space_id: str | None, limit: int, offset: int
    ) -> list[MemoryRecord]:
        sql, params = "SELECT * FROM memories WHERE tenant_id = ?", [tenant_id]
        if space_id is not None:
            sql += " AND space_id = ?"
            params.append(space_id)
        rows = self._query(
            sql + " ORDER BY updated_at DESC LIMIT ? OFFSET ?", (*params, limit, offset)
        )
        return [_to_record(row) for row in rows]

    def versions(self, tenant_id: str, memory_id: UUID) -> list[dict[str, Any]]:
        rows = self._query(
            """SELECT v.* FROM memory_versions v JOIN memories m ON m.id = v.memory_id
               WHERE m.tenant_id = ? AND v.memory_id = ? ORDER BY v.id""",
            (tenant_id, str(memory_id)),
        )
        return [
            {
                "content": row["content"],
                "memory_type": row["memory_type"],
                "importance": row["importance"],
                "confidence": row["confidence"],
                "metadata": json.loads(row["metadata"]),
                "valid_until": row["valid_until"],
                "replaced_at": row["recorded_at"],
            }
            for row in rows
        ]

    def stats(self, tenant_id: str, now: datetime) -> dict[str, Any]:
        rows = self._query(
            "SELECT space_id, memory_type, COUNT(*) AS n FROM memories WHERE tenant_id = ? "
            "GROUP BY space_id, memory_type",
            (tenant_id,),
        )
        by_space: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for row in rows:
            by_space[row["space_id"]] = by_space.get(row["space_id"], 0) + row["n"]
            by_type[row["memory_type"]] = by_type.get(row["memory_type"], 0) + row["n"]
        where, params = _scope_sql(ScopeFilter(tenant_id, list(by_space), now))
        active = self._query(f"SELECT COUNT(*) FROM memories m WHERE {where}", params)[0][0]
        return {
            "total": sum(by_space.values()),
            "active": active if by_space else 0,
            "by_space": by_space,
            "by_type": by_type,
        }


def _scope_sql(scope: ScopeFilter) -> tuple[str, list[Any]]:
    """WHERE clause (on alias ``m``) enforcing tenant, space, type and validity filters."""
    if not scope.space_ids:
        return "0", []
    clauses = [
        "m.tenant_id = ?",
        f"m.space_id IN ({','.join('?' * len(scope.space_ids))})",
    ]
    params: list[Any] = [scope.tenant_id, *scope.space_ids]
    if scope.memory_types:
        clauses.append(f"m.memory_type IN ({','.join('?' * len(scope.memory_types))})")
        params.extend(t.value for t in scope.memory_types)
    if not scope.include_inactive:
        now = _ts(scope.now)
        clauses += [
            "m.superseded_by IS NULL",
            "(m.valid_from IS NULL OR m.valid_from <= ?)",
            "(m.valid_until IS NULL OR m.valid_until > ?)",
        ]
        params += [now, now]
    return " AND ".join(clauses), params


def _to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=UUID(row["id"]),
        tenant_id=row["tenant_id"],
        space_id=row["space_id"],
        content=row["content"],
        memory_type=row["memory_type"],
        source=row["source"],
        created_at=_parse_ts(row["created_at"]),
        updated_at=_parse_ts(row["updated_at"]),
        valid_from=_parse_ts(row["valid_from"]),
        valid_until=_parse_ts(row["valid_until"]),
        importance=row["importance"],
        confidence=row["confidence"],
        superseded_by=UUID(row["superseded_by"]) if row["superseded_by"] else None,
        metadata=json.loads(row["metadata"]),
    )
