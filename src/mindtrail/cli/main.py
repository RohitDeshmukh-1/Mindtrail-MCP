"""``mindtrail`` command line: run the MCP server, set up clients, inspect and manage memories.

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
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from mindtrail import __version__
from mindtrail.core.config import MindtrailConfig
from mindtrail.core.exceptions import MindtrailError
from mindtrail.core.models import MemoryRecord, SearchHit
from mindtrail.core.project import detect_project
from mindtrail.memory.service import MemoryService

CLIENTS = ("claude-code", "cursor", "vscode", "codex")


def _client_setup(client: str) -> str:
    """Configuration snippet for connecting ``client`` to the local stdio server."""
    command, args = "mindtrail", ["serve"]
    if client == "claude-code":
        return "claude mcp add mindtrail --scope user -- mindtrail serve"
    if client == "cursor":
        return "~/.cursor/mcp.json\n" + json.dumps(
            {"mcpServers": {"mindtrail": {"command": command, "args": args}}}, indent=2
        )
    if client == "vscode":
        return ".vscode/mcp.json\n" + json.dumps(
            {"servers": {"mindtrail": {"type": "stdio", "command": command, "args": args}}},
            indent=2,
        )
    if client == "codex":
        return (
            '~/.codex/config.toml\n[mcp_servers.mindtrail]\ncommand = "mindtrail"\nargs = ["serve"]'
        )
    raise ValueError(client)


def _open() -> tuple[MemoryService, MindtrailConfig]:
    config = MindtrailConfig.from_env()
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


def _warm_up(service: MemoryService) -> None:
    try:
        service.warm_up()
    except Exception:
        logging.getLogger(__name__).exception("model warm-up failed; will retry on first use")


# -- commands ------------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    from mindtrail.mcp.server import create_server

    profile = args.tools or os.environ.get("MINDTRAIL_TOOLS", "core")
    if profile not in ("core", "full"):
        print(f"error: MINDTRAIL_TOOLS must be 'core' or 'full', got {profile!r}", file=sys.stderr)
        return 2
    service, _ = _open()
    server = create_server(
        service, project_space=_project_space(), profile="full" if profile == "full" else "core"
    )
    # Load (or, if `mindtrail init` was skipped, download) models while the client is still
    # connecting, so the agent's first recall doesn't stall. Tool calls wait on the same lock.
    threading.Thread(target=_warm_up, args=(service,), daemon=True).start()
    try:
        server.run("stdio")
    finally:
        service.close()
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    service, config = _open()
    try:
        stats = service.stats()
        model = service.embedding_model
        if model.startswith("fastembed") or service.reranker_model:
            print("Preparing models; the first run downloads them...")
            service.warm_up()
        reindexed = service.reindex_embeddings()
    finally:
        service.close()
    print(f"Mindtrail is ready. Data: {config.db_path} ({stats['total']} memories)")
    print(f"Embeddings: {model}" + (f" (indexed {reindexed} memories)" if reindexed else ""))
    print(f"Reranker:   {service.reranker_model or 'none'}")
    if not model.startswith("fastembed"):
        print('Tip: pipx install "mindtrail[semantic]" for recall by meaning, not just words.')
    print()
    for client in [args.client] if args.client else CLIENTS:
        print(f"-- {client} " + "-" * (60 - len(client)))
        print(_client_setup(client) + "\n")
    print('Then ask your agent: "Remember that this project uses conventional commits."')
    print('Open a new session and ask: "How should I write commit messages here?"')
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = MindtrailConfig.from_env()
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
        hint = (
            ""
            if model.startswith("fastembed")
            else " (install mindtrail[semantic] for better recall)"
        )
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


def cmd_bench(args: argparse.Namespace) -> int:
    from mindtrail.embeddings import create_embedder
    from mindtrail.embeddings.rerankers import create_reranker
    from mindtrail.evaluation import load_dataset, run_benchmark
    from mindtrail.evaluation.locomo import fetch_locomo, run_locomo

    config = MindtrailConfig.from_env()
    embedder = create_embedder(args.embedder, cache_dir=config.model_dir)
    reranker = create_reranker(args.reranker, cache_dir=config.model_dir)
    for source in args.dataset:
        if source.startswith("locomo"):
            split = source.partition(":")[2] or "test"
            path = fetch_locomo(config.home / "benchmarks")
            locomo = run_locomo(path, embedder, reranker=reranker, split=split)
            print(locomo.to_markdown() + "\n")
            if args.json:
                path = args.json.with_stem(f"{args.json.stem}-locomo-{split}")
                path.write_text(locomo.model_dump_json(indent=2), encoding="utf-8")
            continue
        report = run_benchmark(load_dataset(source), embedder, k=args.k, reranker=reranker)
        print(report.to_markdown() + "\n")
        if args.failures:
            for result in report.failures():
                print(
                    f"  {result.id:<5} {result.query!r}\n"
                    f"        got {result.retrieved}  want {result.relevant}"
                    + (f"  forbid {result.forbidden}" if result.forbidden else "")
                )
            print()
        if args.json:
            path = (
                args.json
                if len(args.dataset) == 1
                else args.json.with_stem(f"{args.json.stem}-{Path(source).stem}")
            )
            path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return 0


# -- parser --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindtrail", description="Persistent memory for AI agents, over MCP."
    )
    parser.add_argument("--version", action="version", version=f"mindtrail {__version__}")
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

    p = sub.add_parser("bench", help="run the retrieval benchmark")
    p.add_argument(
        "--dataset",
        nargs="+",
        default=["dev", "holdout"],
        help="bundled datasets (dev, holdout, holdout-v2), locomo[:dev|test|all] "
        "(downloaded; CC BY-NC 4.0), or paths to dataset files",
    )
    p.add_argument("--embedder", choices=["auto", "hashing", "fastembed"], default="auto")
    p.add_argument(
        "--reranker", default="auto", help="auto, none, or a fastembed cross-encoder model id"
    )
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--json", type=Path, help="write the full report as JSON")
    p.add_argument("--failures", action="store_true", help="list queries that missed")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("reindex", help="embed memories missing a vector for the current model")
    p.set_defaults(fn=cmd_reindex)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("MINDTRAIL_LOG_LEVEL", "WARNING").upper(),
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        code: int = args.fn(args)
    except (MindtrailError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code
