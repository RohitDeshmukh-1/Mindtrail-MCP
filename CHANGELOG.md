# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Core memory engine: validated immutable memory records, SQLite storage with FTS5, hybrid
  keyword + vector retrieval, token-budgeted context, versioned updates, supersession,
  validity windows, hard/soft deletion and secret filtering on write.
- Offline hashing embedder (default) and optional local neural embeddings via fastembed.
- MCP server over stdio with a three-tool core profile (`remember`, `recall`, `forget`) and a
  `full` profile that adds `search_memory`, `get_context`, `update_memory` and `get_memory`.
- Automatic project scoping from the git remote, shared across tools working in the same repo.
- `mindtrail` CLI: `serve`, `init`, `doctor`, `remember`, `recall`, `forget`, `list`, `stats`,
  `export` and `reindex`.
