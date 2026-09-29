from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from mindtrail.core.models import MemoryRecord, validate_space_id
from mindtrail.core.text import build_fts_query, content_hash
from mindtrail.embeddings import HashingEmbedder
from mindtrail.memory.safety import find_secrets


def test_content_is_stripped_and_blank_rejected() -> None:
    assert MemoryRecord(content="  hello  ").content == "hello"
    with pytest.raises(ValidationError):
        MemoryRecord(content="   ")


def test_naive_timestamps_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(content="x", valid_from=datetime(2026, 1, 1))


def test_validity_window_must_be_ordered() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        MemoryRecord(content="x", valid_from=start, valid_until=start - timedelta(days=1))


def test_records_are_immutable() -> None:
    record = MemoryRecord(content="x")
    with pytest.raises(ValidationError):
        record.content = "y"  # type: ignore[misc]


def test_oversized_metadata_rejected() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(content="x", metadata={"blob": "a" * 5000})


@pytest.mark.parametrize("space", ["Repo:Mindtrail-Core", "project/mindtrail", "default"])
def test_valid_space_ids_are_normalized(space: str) -> None:
    assert validate_space_id(space) == space.lower()


@pytest.mark.parametrize("space", ["", "has space", "../etc", "x" * 200, "tenant'; drop"])
def test_invalid_space_ids_rejected(space: str) -> None:
    with pytest.raises(ValueError):
        validate_space_id(space)


def test_fts_query_neutralizes_operators() -> None:
    query = build_fts_query('"; DROP TABLE memories -- NEAR(a b) AND content:* OR')
    assert query is not None
    assert all(part.startswith('"') and part.endswith('"') for part in query.split(" OR "))


def test_fts_query_none_for_stopwords_only() -> None:
    assert build_fts_query("what is the") is None


def test_content_hash_ignores_case_and_whitespace() -> None:
    assert content_hash("Use  PostgreSQL\n") == content_hash("use postgresql")


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder()
    a = embedder.embed_query("database migrations with alembic")
    b = embedder.embed_query("database migrations with alembic")
    assert (a == b).all()
    assert abs(float((a * a).sum()) - 1.0) < 1e-5
    related = float(a @ embedder.embed_query("alembic migration for the database"))
    unrelated = float(a @ embedder.embed_query("favorite pizza topping"))
    assert related > unrelated


@pytest.mark.parametrize(
    "text",
    [
        "aws key AKIAABCDEFGHIJKLMNOP",
        "token ghp_" + "a" * 36,
        "OPENAI key sk-proj-abcdefghijklmnopqrstuvwx",
        "-----BEGIN RSA PRIVATE KEY-----",
        'password = "hunter2hunter2hunter2"',
    ],
)
def test_secrets_detected(text: str) -> None:
    assert find_secrets(text)


@pytest.mark.parametrize(
    "text",
    [
        "Read the API key from os.environ['API_KEY'], never hardcode it",
        "api_key = os.environ.get('OPENAI_API_KEY')",
        "The user prefers tabs over spaces",
        "Rotate the password every 90 days",
    ],
)
def test_ordinary_text_not_flagged(text: str) -> None:
    assert find_secrets(text) == []
