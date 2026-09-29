from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from tests.conftest import FakeClock

from mindtrail.core.exceptions import InvalidMemoryError, MemoryNotFoundError, SecretDetectedError
from mindtrail.core.models import MemoryType
from mindtrail.embeddings import HashingEmbedder
from mindtrail.memory.service import MemoryService
from mindtrail.storage.sqlite import SQLiteMemoryRepository


def _contents(hits: list) -> list[str]:  # type: ignore[type-arg]
    return [hit.memory.content for hit in hits]


# -- remember / get ------------------------------------------------------------------------


def test_remember_and_get_roundtrip(service: MemoryService) -> None:
    result = service.remember(
        "The project uses PostgreSQL with pgvector",
        memory_type="semantic",
        source="claude-code",
        importance=0.8,
        metadata={"repo": "mindtrail"},
    )
    assert not result.deduplicated
    fetched = service.get(result.memory.id)
    assert fetched == result.memory
    assert fetched.metadata == {"repo": "mindtrail"}


def test_exact_duplicates_are_merged(service: MemoryService, clock: FakeClock) -> None:
    first = service.remember("User prefers tabs over spaces", importance=0.4)
    clock.advance(hours=1)
    second = service.remember("  user prefers TABS over spaces ", importance=0.9)
    assert second.deduplicated
    assert second.memory.id == first.memory.id
    assert second.memory.importance == 0.9
    assert service.stats()["total"] == 1


def test_same_content_in_different_spaces_is_not_merged(service: MemoryService) -> None:
    service.remember("Use ruff for linting", space_id="repo:a")
    result = service.remember("Use ruff for linting", space_id="repo:b")
    assert not result.deduplicated


def test_secrets_are_refused(service: MemoryService) -> None:
    with pytest.raises(SecretDetectedError):
        service.remember("my github token is ghp_" + "x" * 36)
    with pytest.raises(SecretDetectedError):
        service.remember("deploy notes", metadata={"key": "AKIAABCDEFGHIJKLMNOP"})
    assert service.stats()["total"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": ""},
        {"content": "x", "importance": 2.0},
        {"content": "x", "memory_type": "nonsense"},
        {"content": "x", "space_id": "bad space"},
    ],
)
def test_invalid_input_raises_clean_error(service: MemoryService, kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(InvalidMemoryError):
        service.remember(**kwargs)


def test_get_unknown_and_malformed_ids(service: MemoryService) -> None:
    with pytest.raises(MemoryNotFoundError):
        service.get(uuid4())
    with pytest.raises(InvalidMemoryError):
        service.get("not-a-uuid")


# -- search / recall -----------------------------------------------------------------------


@pytest.fixture
def seeded(service: MemoryService) -> MemoryService:
    for content in [
        "The backend uses FastAPI and PostgreSQL",
        "User prefers pytest over unittest for testing",
        "Deployment runs on Docker containers in Fly.io",
        "The user's favorite pizza topping is mushrooms",
        "Database migrations are managed with Alembic",
    ]:
        service.remember(content)
    return service


def test_recall_ranks_relevant_memory_first(seeded: MemoryService) -> None:
    assert _contents(seeded.recall("which test framework does the user like?"))[0] == (
        "User prefers pytest over unittest for testing"
    )
    assert _contents(seeded.recall("how are database migrations handled"))[0] == (
        "Database migrations are managed with Alembic"
    )


def test_recall_returns_empty_when_nothing_relevant(seeded: MemoryService) -> None:
    assert seeded.recall("quantum chromodynamics lattice") == []


def test_hits_carry_explainable_signals(seeded: MemoryService) -> None:
    hit = seeded.recall("FastAPI backend")[0]
    assert {"relevance", "recency", "keyword_rank"} <= hit.signals.keys()
    assert hit.score > 0


def test_search_survives_hostile_query_syntax(seeded: MemoryService) -> None:
    seeded.search('"; DROP TABLE memories; -- NEAR( * content:')
    with pytest.raises(InvalidMemoryError):
        seeded.search("   ")


def test_search_filters_by_memory_type(service: MemoryService) -> None:
    service.remember("Fixed the login bug on Monday", memory_type=MemoryType.EPISODIC)
    service.remember("Login uses OAuth", memory_type=MemoryType.SEMANTIC)
    hits = service.search("login", memory_types=["episodic"])
    assert _contents(hits) == ["Fixed the login bug on Monday"]


def test_importance_breaks_ties_between_equally_relevant(service: MemoryService) -> None:
    service.remember("Service alpha uses Redis", importance=0.1)
    service.remember("Service beta uses Redis", importance=0.9)
    assert _contents(service.recall("Redis"))[0] == "Service beta uses Redis"


# -- spaces and tenants --------------------------------------------------------------------


def test_spaces_are_isolated(service: MemoryService) -> None:
    service.remember("Uses tabs for indentation", space_id="repo:frontend")
    service.remember("Uses spaces for indentation", space_id="repo:backend")
    assert _contents(service.recall("indentation", space_id="repo:frontend")) == [
        "Uses tabs for indentation"
    ]
    assert service.recall("indentation") == []  # personal space holds nothing
    both = service.search("indentation", space_ids=["repo:frontend", "repo:backend"])
    assert len(both) == 2


def test_tenants_cannot_see_each_other(repo: SQLiteMemoryRepository, clock: FakeClock) -> None:
    alice = MemoryService(repo, HashingEmbedder(), tenant_id="alice", clock=clock)
    bob = MemoryService(repo, HashingEmbedder(), tenant_id="bob", clock=clock)
    secret = alice.remember("Alice's launch date is October 3").memory

    assert bob.recall("launch date October") == []
    with pytest.raises(MemoryNotFoundError):
        bob.get(secret.id)
    with pytest.raises(MemoryNotFoundError):
        bob.forget(secret.id)
    with pytest.raises(MemoryNotFoundError):
        bob.update_memory(secret.id, content="hijacked")
    assert bob.stats()["total"] == 0
    assert alice.get(secret.id).content == "Alice's launch date is October 3"


# -- update / supersede / forget / validity ------------------------------------------------


def test_update_keeps_history_and_reindexes(service: MemoryService, clock: FakeClock) -> None:
    memory = service.remember("Deadline is October 1").memory
    clock.advance(days=1)
    updated = service.update_memory(memory.id, content="Deadline is October 15", importance=0.9)

    assert updated.id == memory.id and updated.updated_at > memory.updated_at
    assert [v["content"] for v in service.history(memory.id)] == ["Deadline is October 1"]
    assert _contents(service.recall("October 15 deadline"))[0] == "Deadline is October 15"
    assert "Deadline is October 1" not in _contents(service.search("deadline October"))


def test_update_rejects_secrets_and_invalid_values(service: MemoryService) -> None:
    memory = service.remember("CI config lives in .github").memory
    with pytest.raises(SecretDetectedError):
        service.update_memory(memory.id, content="token: ghp_" + "y" * 36)
    with pytest.raises(InvalidMemoryError):
        service.update_memory(memory.id, importance=-1)
    assert service.get(memory.id).content == "CI config lives in .github"


def test_supersede_hides_old_version(service: MemoryService) -> None:
    old = service.remember("Project deadline is October 1").memory
    result = service.remember("Project deadline is October 15", supersedes=old.id)

    assert result.superseded_id == old.id
    assert _contents(service.recall("project deadline")) == ["Project deadline is October 15"]
    assert service.get(old.id).superseded_by == result.memory.id
    everything = service.search("project deadline", include_inactive=True)
    assert len(everything) == 2


def test_hard_forget_deletes_everywhere(service: MemoryService) -> None:
    memory = service.remember("Temporary note about staging credentials rotation").memory
    service.update_memory(memory.id, importance=0.9)  # creates a version row too
    assert service.forget(memory.id) is True
    with pytest.raises(MemoryNotFoundError):
        service.get(memory.id)
    assert service.recall("staging credentials rotation") == []
    assert service.stats()["total"] == 0


def test_soft_forget_invalidates_but_keeps_record(service: MemoryService, clock: FakeClock) -> None:
    memory = service.remember("Team standup is at 9am").memory
    clock.advance(minutes=5)
    service.forget(memory.id, hard=False)
    assert service.recall("standup time") == []
    assert service.get(memory.id).valid_until == clock.now
    assert service.remember("Team standup is at 9am").deduplicated is False


def test_validity_window_controls_visibility(service: MemoryService, clock: FakeClock) -> None:
    service.remember(
        "Office closed for renovation",
        valid_from=clock.now + timedelta(days=1),
        valid_until=clock.now + timedelta(days=3),
    )
    assert service.recall("office renovation") == []
    clock.advance(days=2)
    assert _contents(service.recall("office renovation")) == ["Office closed for renovation"]
    clock.advance(days=2)
    assert service.recall("office renovation") == []


# -- context, persistence, reindex ---------------------------------------------------------


def test_get_context_is_budgeted_and_empty_when_irrelevant(seeded: MemoryService) -> None:
    ctx = seeded.get_context("what database and migration tools are used?", token_budget=300)
    assert ctx.memories and ctx.token_estimate <= 300
    assert "Alembic" in ctx.text
    assert seeded.get_context("quantum chromodynamics").text == ""


def test_memories_survive_restart(tmp_path: Path) -> None:
    db = tmp_path / "mindtrail.db"
    first = MemoryService(SQLiteMemoryRepository(db), HashingEmbedder())
    memory = first.remember("Repository convention: use conventional commits").memory
    first.close()

    second = MemoryService(SQLiteMemoryRepository(db), HashingEmbedder())
    try:
        assert second.get(memory.id).content == memory.content
        assert _contents(second.recall("commit convention")) == [memory.content]
    finally:
        second.close()


def test_reindex_after_embedding_model_change(repo: SQLiteMemoryRepository) -> None:
    MemoryService(repo, HashingEmbedder(dimension=256)).remember("Frontend is built with React")
    upgraded = MemoryService(repo, HashingEmbedder(dimension=512))
    assert upgraded.reindex_embeddings() == 1
    assert upgraded.reindex_embeddings() == 0
    hit = upgraded.recall("React frontend")[0]
    assert "vector_similarity" in hit.signals


def test_embedding_failure_does_not_lose_write(repo: SQLiteMemoryRepository) -> None:
    class BrokenEmbedder(HashingEmbedder):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            raise RuntimeError("model unavailable")

    service = MemoryService(repo, BrokenEmbedder())
    memory = service.remember("Keyword search still works without vectors").memory
    assert _contents(service.recall("keyword search vectors")) == [memory.content]


def test_list_and_stats(service: MemoryService, clock: FakeClock) -> None:
    service.remember("First fact", space_id="repo:a")
    clock.advance(minutes=1)
    service.remember("Second fact", space_id="repo:a", memory_type="episodic")
    expired = service.remember("Third fact", space_id="repo:b").memory
    service.forget(expired.id, hard=False)

    assert [m.content for m in service.list_memories(space_id="repo:a")] == [
        "Second fact",
        "First fact",
    ]
    stats = service.stats()
    assert stats["total"] == 3 and stats["active"] == 2
    assert stats["by_space"] == {"repo:a": 2, "repo:b": 1}
    assert stats["by_type"] == {"semantic": 2, "episodic": 1}
