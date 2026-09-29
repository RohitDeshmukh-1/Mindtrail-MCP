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
  `MINDTRAIL_PROJECT`.
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
- Vector search is a brute-force scan (fine up to tens of thousands of memories per space).
- Project detection uses the server's working directory; MCP roots are not used yet.
- There is no contradiction detection between different active memories.

## ✅ Milestone 3 — Retrieval evaluation and calibration

- `mindtrail bench` runs two bundled datasets: `dev` (57 memories, 92 queries) and a
  held-out set (30 memories, 39 queries) written after tuning. Metrics: recall@1/5, MRR,
  nDCG@5, abstention, stale/foreign leak rate, p50/p95 latency.
- Calibrated candidate gates (BM25 relative floor, vector support for keyword hits,
  per-embedder similarity floors). On holdout with bge-small, abstention is 91% and recall@5
  is 89%.
- The `[semantic]` extra (fastembed + bge-small, 65 MB) stores its model in
  `~/.mindtrail/models`; `mindtrail init` pre-downloads it so the first MCP call doesn't stall.
- CI gates: zero stale/foreign leaks, 100% recall@5 on wording-matched queries, and overall
  floors just under measured values. A separate CI job runs the neural benchmark and posts
  the table to the job summary.
- Results and methodology: [benchmarks/README.md](benchmarks/README.md).

**Verification:** 90 tests pass locally (Windows, Python 3.13), including the neural gates with
fastembed 0.8.1. `ruff` and `mypy --strict` pass.

**Known limitations**
- The benchmark is small, synthetic and written by the authors. There is no agent-in-the-loop
  evaluation yet.
- The bge-small thresholds are sensitive: a 0.05 step changes abstention sharply.
- The default install (hashing) is weak on paraphrase (46% recall@5 on holdout).

## Next — Milestone 4: agent-in-the-loop evaluation
- Scripted multi-session tasks run against a real agent, with and without Mindtrail: does the
  agent call `recall` and `remember` unprompted, and does task success improve?
- A decision, informed by that, on whether `[semantic]` should become the default install.
- Use MCP roots for project detection instead of the server's working directory.

## Later
PostgreSQL + pgvector with multi-tenancy, then hosted remote MCP with OAuth and a web memory
viewer, then graph and consolidation features.
