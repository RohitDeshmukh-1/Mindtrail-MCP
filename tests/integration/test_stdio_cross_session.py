"""The Phase 1 exit criterion: store in one agent session, recall in a separate one.

Each session launches `python -m cogmem serve` as a real subprocess over stdio, the same way
Claude Code, Cursor or Codex launch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _server(home: Path, project: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "cogmem", "serve"],
        env={"COGMEM_HOME": str(home), "COGMEM_EMBEDDER": "hashing", "COGMEM_PROJECT": project},
    )


async def test_memory_survives_across_separate_server_processes(tmp_path: Path) -> None:
    async with Client(_server(tmp_path, "demo"), read_timeout_seconds=60) as first_session:
        stored = await first_session.call_tool(
            "remember", {"content": "Database migrations are run with `alembic upgrade head`"}
        )
        assert not stored.is_error

    async with Client(_server(tmp_path, "demo"), read_timeout_seconds=60) as second_session:
        recalled = await second_session.call_tool("recall", {"query": "how do I run migrations"})

    assert recalled.structured_content is not None
    contents = [m["content"] for m in recalled.structured_content["memories"]]
    assert contents == ["Database migrations are run with `alembic upgrade head`"]

    async with Client(_server(tmp_path, "unrelated"), read_timeout_seconds=60) as elsewhere:
        isolated = await elsewhere.call_tool("recall", {"query": "migrations"})
    assert isolated.structured_content == {
        "note": recalled.structured_content["note"],
        "memories": [],
    }
