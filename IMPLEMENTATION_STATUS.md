# Implementation Status

Roadmap: [docs/design/specification.md](docs/design/specification.md) (full spec). Each
milestone is tested before the next one starts.

## ✅ Milestone 1 — Core memory engine

A transport-independent engine (`MemoryService`) that every interface calls.

- Immutable, validated `MemoryRecord` (UTC timestamps, validity windows, bounded content and
  metadata, normalized space ids).
- SQLite backend: FTS5 keyword index, float32 vectors, version history, `secure_delete`, WAL,
  migrations. Every query is tenant-bound.
- Embeddings: offline hashing embedder (default) and optional fastembed (`bge-small-en-v1.5`).
- Hybrid ranking (RRF plus importance, confidence and recency boosts), with the signals returned
  alongside each result.
- Token-budgeted context inside an untrusted-data boundary that stored content cannot escape.
- Write-path secret filter; dedup, supersede, hard and soft forget, reindex.

## ✅ Milestone 2 — MCP server and CLI

- MCP server (official `mcp` SDK 2.x, stdio). The `core` profile has remember, recall and
  forget; the `full` profile adds search_memory, get_context, update_memory and get_memory.
  Server instructions tell agents when to use the tools. Engine errors reach the model as tool
  errors.
- Automatic project scoping: the space id comes from the normalized git remote (credentials
  stripped), so the same repo shares memory across tools and clones. Override with
  `COGMEM_PROJECT`.
- CLI: `serve`, `init`, `doctor`, `remember`, `recall`, `forget`, `list`, `stats`, `export`,
  `reindex`.
- Repository setup: MIT license, CI (Linux, macOS and Windows × Python 3.11–3.13, plus lint,
  types and build), PyPI trusted-publishing release workflow, issue and PR templates,
  Dependabot, pre-commit, contributing, security and conduct docs.

**Verification**
- 81 automated tests pass locally on Windows with Python 3.13: unit tests, in-process MCP client
  tests, and a real two-process stdio test (store in one server process, recall in a new one,
  and confirm another project can't see it).
- `ruff check`, `ruff format --check` and `mypy --strict` pass. The sdist and wheel build, and
  `twine check` passes.
- Claude Code: `claude mcp add` followed by `claude mcp list` reports **Connected**. An agent
  actually choosing to call the tools during a real session has not been measured yet.
- Cursor, VS Code and Codex configs are documented but **not verified**.
- The CI workflow has not run yet because the repo is not on GitHub.

**Known limitations**
- The PyPI name `cogmem` is taken by an unrelated project. The distribution is `cogmem-mcp`
  for now; a final name is pending.
- Vector search is a brute-force scan (fine up to tens of thousands of memories per space).
- The hashing embedder is lexical only. The fastembed path has no automated tests.
- Project detection uses the server's working directory; MCP roots are not used yet.
- There is no contradiction detection between different active memories.

## Next — Milestone 3: retrieval evaluation and semantic quality
- A synthetic benchmark (about 100 cases: recall, paraphrase, temporal, contradiction) with
  recall@k and MRR, run in CI.
- Test coverage for the fastembed path, and a measured comparison against the hashing embedder.
- An agent-behavior check: does an agent call `recall` and `remember` without being prompted?

## Later
PostgreSQL + pgvector with multi-tenancy, then hosted remote MCP with OAuth and a web memory
viewer, then graph and consolidation features.
