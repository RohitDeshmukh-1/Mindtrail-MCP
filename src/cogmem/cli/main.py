"""``cogmem`` command line: run the MCP server, set up clients, inspect and manage memories.

stdout is reserved for the MCP protocol while ``serve`` runs; all logs go to stderr.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from cogmem import __version__
from cogmem.core.config import CogMemConfig
from cogmem.core.exceptions import CogMemError
from cogmem.core.models import MemoryRecord, SearchHit
from cogmem.core.project import detect_project
from cogmem.memory.service import MemoryService

CLIENTS = ("claude-code", "cursor", "vscode", "codex")


def _client_setup(client: str) -> str:
    """Configuration snippet for connecting ``client`` to the local stdio server."""
    command, args = "cogmem", ["serve"]
    if client == "claude-code":
        return "claude mcp add cogmem --scope user -- cogmem serve"
    if client == "cursor":
        return "~/.cursor/mcp.json\n" + json.dumps(
            {"mcpServers": {"cogmem": {"command": command, "args": args}}}, indent=2
        )
    if client == "vscode":
        return ".vscode/mcp.json\n" + json.dumps(
            {"servers": {"cogmem": {"type": "stdio", "command": command, "args": args}}}, indent=2
        )
    if client == "codex":
        return '~/.codex/config.toml\n[mcp_servers.cogmem]\ncommand = "cogmem"\nargs = ["serve"]'
    raise ValueError(client)


def _open() -> tuple[MemoryService, CogMemConfig]:
    config = CogMemConfig.from_env()
    return MemoryService.from_config(config), config


def _project_space() -> str | None:
    project = detect_project()
    return project.space_id if project else None


def _print_hits(hits: Sequence[SearchHit], as_json: bool) -> None:
    if as_json:
        print(json.dumps([hit.model_dump(mode="json") for hit in hits], indent=2))
        return
    if not hits:
        print("No relevant memories.")
    for hit in hits:
        memory = hit.memory
        print(f"{hit.score:5.2f}  {memory.content}")
        print(f"       id={memory.id}  space={memory.space_id}  type={memory.memory_type.value}")


def _print_records(records: Sequence[MemoryRecord], as_json: bool) -> None:
    if as_json:
        print(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
        return
    if not records:
        print("No memories.")
    for record in records:
        print(f"{record.updated_at:%Y-%m-%d}  [{record.space_id}] {record.content}")
        print(f"            id={record.id}")


# -- commands ------------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    from cogmem.mcp.server import create_server

    profile = args.tools or os.environ.get("COGMEM_TOOLS", "core")
    if profile not in ("core", "full"):
        print(f"error: COGMEM_TOOLS must be 'core' or 'full', got {profile!r}", file=sys.stderr)
        return 2
    service, _ = _open()
    server = create_server(
        service, project_space=_project_space(), profile="full" if profile == "full" else "core"
    )
    try:
        server.run("stdio")
    finally:
        service.close()
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    service, config = _open()
    try:
        stats = service.stats()
    finally:
        service.close()
    print(f"CogMem is ready. Data: {config.db_path} ({stats['total']} memories)\n")
    for client in [args.client] if args.client else CLIENTS:
        print(f"-- {client} " + "-" * (60 - len(client)))
        print(_client_setup(client) + "\n")
    print('Then ask your agent: "Remember that this project uses conventional commits."')
    print('Open a new session and ask: "How should I write commit messages here?"')
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = CogMemConfig.from_env()
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn: Callable[[], str]) -> None:
        try:
            checks.append((name, True, fn()))
        except Exception as exc:
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    def python_version() -> str:
        return sys.version.split()[0]

    def fts5() -> str:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
        return f"SQLite {sqlite3.sqlite_version} with FTS5"

    def database() -> str:
        service = MemoryService.from_config(config)
        try:
            stats = service.stats()
            return f"{config.db_path} ({stats['total']} memories, {stats['active']} active)"
        finally:
            service.close()

    def embedder() -> str:
        service = MemoryService.from_config(config)
        try:
            service.search("doctor check")
            model = service.embedding_model
        finally:
            service.close()
        hint = "" if model.startswith("fastembed") else " (install [local-embeddings] for semantic)"
        return model + hint

    def mcp_sdk() -> str:
        if importlib.util.find_spec("mcp") is None:
            raise RuntimeError("the 'mcp' package is not installed")
        from importlib.metadata import version

        return f"mcp {version('mcp')}"

    def project() -> str:
        found = detect_project()
        return found.space_id if found else "none detected (memories default to personal)"

    check("python", python_version)
    check("sqlite", fts5)
    check("database", database)
    check("embeddings", embedder)
    check("mcp sdk", mcp_sdk)
    check("project", project)

    for name, ok, detail in checks:
        print(f"{'ok ' if ok else 'ERR'}  {name:<11} {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def cmd_remember(args: argparse.Namespace) -> int:
    service, _ = _open()
    try:
        project = _project_space() if args.scope == "project" else None
        result = service.remember(
            args.content,
            memory_type=args.type,
            space_id=project or service.default_space,
            importance=args.importance,
            source="cli",
        )
    finally:
        service.close()
    verb = "Already known" if result.deduplicated else "Remembered"
    print(f"{verb}: {result.memory.id} [{result.memory.space_id}]")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    service, _ = _open()
    try:
        spaces = [s for s in (_project_space(), service.default_space) if s]
        _print_hits(service.search(args.query, space_ids=spaces, limit=args.limit), args.json)
    finally:
        service.close()
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    service, _ = _open()
    try:
        service.forget(args.memory_id, hard=not args.soft)
    finally:
        service.close()
    print(("Invalidated " if args.soft else "Deleted ") + args.memory_id)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    service, _ = _open()
    try:
        _print_records(
            service.list_memories(space_id=args.space, limit=args.limit, offset=args.offset),
            args.json,
        )
    finally:
        service.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    service, _ = _open()
    try:
        print(json.dumps(service.stats(), indent=2))
    finally:
        service.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    service, _ = _open()
    count = 0
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout  # noqa: SIM115
    try:
        offset = 0
        while batch := service.list_memories(limit=500, offset=offset):
            for record in batch:
                out.write(record.model_dump_json() + "\n")
            count += len(batch)
            offset += len(batch)
    finally:
        service.close()
        if out is not sys.stdout:
            out.close()
    print(f"Exported {count} memories.", file=sys.stderr)
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    service, _ = _open()
    try:
        count = service.reindex_embeddings()
        model = service.embedding_model
    finally:
        service.close()
    print(f"Embedded {count} memories with {model}.")
    return 0


# -- parser --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cogmem", description="Persistent memory for AI agents, over MCP."
    )
    parser.add_argument("--version", action="version", version=f"cogmem {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("serve", help="run the MCP server over stdio")
    p.add_argument("--tools", choices=["core", "full"], help="tool profile (default: core)")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("init", help="create the local store and print client setup")
    p.add_argument("--client", choices=CLIENTS)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("doctor", help="check the installation")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("remember", help="store a memory")
    p.add_argument("content")
    p.add_argument("--scope", choices=["project", "personal"], default="project")
    p.add_argument("--type", default="semantic")
    p.add_argument("--importance", type=float, default=0.5)
    p.set_defaults(fn=cmd_remember)

    p = sub.add_parser("recall", help="search project and personal memories")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_recall)

    p = sub.add_parser("forget", help="delete a memory")
    p.add_argument("memory_id")
    p.add_argument("--soft", action="store_true", help="invalidate but keep history")
    p.set_defaults(fn=cmd_forget)

    p = sub.add_parser("list", help="list memories, newest first")
    p.add_argument("--space")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("stats", help="memory counts by space and type")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("export", help="export all memories as JSON Lines")
    p.add_argument("-o", "--output", type=Path)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("reindex", help="embed memories missing a vector for the current model")
    p.set_defaults(fn=cmd_reindex)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("COGMEM_LOG_LEVEL", "WARNING").upper(),
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        code: int = args.fn(args)
    except (CogMemError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code
