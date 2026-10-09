# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-09

First public release on PyPI and the MCP Registry.

### Changed
- `mindtrail[semantic]` now uses `BAAI/bge-base-en-v1.5` (210 MB) by default. It is ahead on
  every bundled set (holdout v2: 94% recall@5, 100% abstention). Choose another model with
  `MINDTRAIL_EMBEDDING_MODEL`, for example `BAAI/bge-small-en-v1.5` (67 MB). Existing memories
  are re-embedded automatically when the server starts.
- Cross-encoder reranking is off by default; enable it with `MINDTRAIL_RERANKER=<model id>`.
- Server instructions now tell agents to call `recall` before any project-specific answer,
  even a one-line one. In the agent evaluation this fixed the cases where the agent answered
  from habit (`npm install`, `str | None` on a Python 3.9 project) instead of checking memory.
- Renamed the project from CogMem to **Mindtrail** (package, CLI, `MINDTRAIL_*` variables,
  `~/.mindtrail`).
- The `local-embeddings` extra is now `semantic`. Neural models are stored in
  `~/.mindtrail/models`, and `mindtrail init` pre-downloads them.
- Retrieval now drops weak keyword matches (below 40% of the best BM25 score, or without vector
  support) and uses calibrated similarity floors per embedder, so a query with no stored answer
  is far more likely to return nothing.

### Fixed
- Clones of the same repository in differently named folders now share one project space.
  The space name came from the folder instead of the git remote. In a repository whose folder
  name differs from its remote's repo name, memories stored before this fix stay under the
  old space id.

### Added
- Agent-in-the-loop evaluation (`benchmarks/agent_eval.py`): real Claude Code sessions,
  with and without Mindtrail.
- `holdout-v2` retrieval set and LoCoMo (`mindtrail bench --dataset locomo:test`), with
  bootstrap confidence intervals in every report.
- Retrieval benchmark (`mindtrail bench`) with bundled dev and held-out datasets: recall@k,
  MRR, nDCG, abstention, stale/foreign leak rate and latency. CI gates on the results.
- Core memory engine: validated immutable memory records, SQLite storage with FTS5, hybrid
  keyword + vector retrieval, token-budgeted context, versioned updates, supersession,
  validity windows, hard/soft deletion and secret filtering on write.
- Offline hashing embedder (default) and optional local neural embeddings via fastembed.
- MCP server over stdio with a three-tool core profile (`remember`, `recall`, `forget`) and a
  `full` profile that adds `search_memory`, `get_context`, `update_memory` and `get_memory`.
- Automatic project scoping from the git remote, shared across tools working in the same repo.
- `mindtrail` CLI: `serve`, `init`, `doctor`, `remember`, `recall`, `forget`, `list`, `stats`,
  `export` and `reindex`.
