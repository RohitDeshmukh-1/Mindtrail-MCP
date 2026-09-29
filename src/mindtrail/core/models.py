"""Domain models shared by every Mindtrail interface (MCP, REST, SDK, CLI)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CONTENT_CHARS = 8_000
MAX_METADATA_BYTES = 4_000
DEFAULT_SPACE = "personal"
_SPACE_ID = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,127}$")


def utcnow() -> datetime:
    return datetime.now(UTC)


def validate_space_id(value: str) -> str:
    value = value.strip().lower()
    if not _SPACE_ID.match(value):
        raise ValueError(
            "space_id must be 1-128 characters of lowercase letters, digits and . _ : / -"
        )
    return value


class MemoryType(StrEnum):
    SEMANTIC = "semantic"  # stable facts and preferences
    EPISODIC = "episodic"  # events and interactions
    TEMPORAL = "temporal"  # facts with a validity period
    REFLECTIVE = "reflective"  # consolidated patterns and insights


class MemoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str = "local"
    space_id: str = DEFAULT_SPACE

    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    memory_type: MemoryType = MemoryType.SEMANTIC
    source: str | None = Field(default=None, max_length=200)

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    superseded_by: UUID | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value

    @field_validator("space_id")
    @classmethod
    def _check_space(cls, value: str) -> str:
        return validate_space_id(value)

    @field_validator("created_at", "updated_at", "valid_from", "valid_until")
    @classmethod
    def _require_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("metadata")
    @classmethod
    def _check_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        if len(encoded.encode()) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata must be at most {MAX_METADATA_BYTES} bytes as JSON")
        return value

    @model_validator(mode="after")
    def _check_validity_window(self) -> MemoryRecord:
        if self.valid_from and self.valid_until and self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be after valid_from")
        return self

    def is_active(self, at: datetime) -> bool:
        """Not superseded and inside its validity window at time ``at``."""
        if self.superseded_by is not None:
            return False
        if self.valid_from is not None and self.valid_from > at:
            return False
        return self.valid_until is None or self.valid_until > at


class RememberResult(BaseModel):
    memory: MemoryRecord
    deduplicated: bool = False
    superseded_id: UUID | None = None


class SearchHit(BaseModel):
    memory: MemoryRecord
    score: float
    signals: dict[str, float] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    query: str
    text: str
    memories: list[SearchHit]
    token_estimate: int
    omitted: int = 0
