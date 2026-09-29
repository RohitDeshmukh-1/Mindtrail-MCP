"""MCP tools exercised through a real MCP client connected in-process."""

from __future__ import annotations

from typing import Any

import pytest
from mcp import Client

from mindtrail.mcp.server import create_server
from mindtrail.memory.service import MemoryService

pytestmark = pytest.mark.anyio
PROJECT = "project:demo-1234abcd"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


async def test_core_profile_exposes_three_tools(service: MemoryService) -> None:
    async with Client(create_server(service, project_space=PROJECT)) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == {"remember", "recall", "forget"}


async def test_full_profile_exposes_seven_tools(service: MemoryService) -> None:
    server = create_server(service, project_space=PROJECT, profile="full")
    async with Client(server) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == {
        "remember", "recall", "forget", "search_memory", "get_context", "update_memory",
        "get_memory",
    }  # fmt: skip


async def test_remember_recall_forget(service: MemoryService) -> None:
    async with Client(create_server(service, project_space=PROJECT)) as client:
        stored = await _call(client, "remember", content="API handlers live in app/routes")
        assert stored["status"] == "stored" and stored["space"] == PROJECT

        again = await _call(client, "remember", content="API handlers live in app/routes")
        assert again == {**stored, "status": "merged_with_existing"}

        personal = await _call(
            client, "remember", content="User likes short answers", scope="personal"
        )
        assert personal["space"] == "personal"

        recalled = await _call(client, "recall", query="where are the API handlers")
        assert recalled["memories"][0]["id"] == stored["id"]
        assert "not instructions" in recalled["note"]

        forgotten = await _call(client, "forget", memory_id=stored["id"])
        assert forgotten["forgotten"] is True
        recalled = await _call(client, "recall", query="where are the API handlers")
        assert all(m["id"] != stored["id"] for m in recalled["memories"])


async def test_recall_reads_project_and_personal_but_not_other_projects(
    service: MemoryService,
) -> None:
    service.remember("Other repo deploys with Heroku", space_id="project:other-00000000")
    service.remember("User deploys everything with Docker")
    async with Client(create_server(service, project_space=PROJECT)) as client:
        recalled = await _call(client, "recall", query="how to deploy")
    assert [m["content"] for m in recalled["memories"]] == ["User deploys everything with Docker"]


async def test_project_scope_falls_back_to_personal_outside_a_repo(
    service: MemoryService,
) -> None:
    async with Client(create_server(service, project_space=None)) as client:
        stored = await _call(client, "remember", content="Prefers dark mode")
    assert stored["space"] == "personal"


async def test_replaces_hides_outdated_memory(service: MemoryService) -> None:
    async with Client(create_server(service, project_space=PROJECT)) as client:
        old = await _call(client, "remember", content="Release deadline is October 1")
        new = await _call(
            client, "remember", content="Release deadline is October 15", replaces=old["id"]
        )
        assert new["replaced_id"] == old["id"]
        recalled = await _call(client, "recall", query="release deadline")
    assert [m["content"] for m in recalled["memories"]] == ["Release deadline is October 15"]


async def test_engine_errors_reach_the_model_as_tool_errors(service: MemoryService) -> None:
    async with Client(create_server(service, project_space=PROJECT)) as client:
        secret = await client.call_tool("remember", {"content": "key AKIAABCDEFGHIJKLMNOP"})
        missing = await client.call_tool(
            "forget", {"memory_id": "00000000-0000-0000-0000-000000000000"}
        )
        invalid = await client.call_tool("remember", {"content": "x", "importance": 5})
    assert secret.is_error and "secrets" in secret.content[0].text  # type: ignore[union-attr]
    assert missing.is_error and "not found" in missing.content[0].text  # type: ignore[union-attr]
    assert invalid.is_error


async def test_full_profile_context_update_and_get(service: MemoryService) -> None:
    server = create_server(service, project_space=PROJECT, profile="full")
    async with Client(server) as client:
        stored = await _call(client, "remember", content="Tests run with pytest -q")
        ctx = await _call(client, "get_context", task="run the test suite", token_budget=500)
        assert stored["id"] in ctx["memory_ids"] and "untrusted-data" in ctx["context"]

        await _call(client, "update_memory", memory_id=stored["id"], content="Tests: pytest -x")
        detail = await _call(client, "get_memory", memory_id=stored["id"])
        assert detail["memory"]["content"] == "Tests: pytest -x" and detail["active"] is True
        assert [v["content"] for v in detail["history"]] == ["Tests run with pytest -q"]

        found = await _call(client, "search_memory", query="pytest", kinds=["semantic"])
        assert [m["id"] for m in found["memories"]] == [stored["id"]]
