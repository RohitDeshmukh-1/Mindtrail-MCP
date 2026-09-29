<div align="center">

# 🧠 CogMem

**Persistent memory for AI coding agents. Tell your agent something once, and every future
session remembers it.**

[![CI](https://github.com/OWNER/cogmem/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/cogmem/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

Every new agent session starts from zero. You explain your conventions again, re-state your
preferences, and re-describe decisions you made last week. **CogMem gives your agents a shared,
long-term memory** through the [Model Context Protocol](https://modelcontextprotocol.io), so
what one session learns, every later session (in any tool) can recall.

- 🔌 **Works with any MCP client.** One memory shared by Claude Code, Cursor, VS Code, Codex
  and your own agents.
- 🏠 **Local-first and private.** One SQLite file on your machine. No account, no API key, no
  telemetry.
- ⚡ **Installs in seconds.** No model download is required; add neural embeddings later with
  one extra.
- 🗂️ **Scoped automatically.** Repo facts stay with the repo (detected from the git remote) and
  personal preferences follow you everywhere.
- 🔎 **Hybrid retrieval.** Keyword (BM25) and vector search, fused and ranked, with the ranking
  signals shown for every result.
- 🛡️ **Safe by default.** Refuses to store credentials, marks recalled memory as untrusted data,
  and deletes for real.

## Quickstart

```bash
pipx install cogmem-mcp                               # or: uv tool install cogmem-mcp
claude mcp add cogmem --scope user -- cogmem serve    # Claude Code
```

Cursor, VS Code, Codex and custom clients are covered in
**[docs/integrations.md](docs/integrations.md)**, or run `cogmem init` to print each config.

Then try it:

```text
Session 1 › Remember that this project uses conventional commits and squash merges.
Session 2 › Write a commit message for these changes.
            → the agent recalls the convention and writes "feat(api): add pagination to /orders"
```

## How it works

Your agent gets three tools, and CogMem's server instructions tell it when to use them:

| Tool | What it does |
|---|---|
| `remember` | Store one fact, preference, decision or event, scoped to the **project** or **personal** space. Pass `replaces` to supersede an outdated memory. |
| `recall` | Find relevant memories from the current project plus your personal space. Returns nothing when nothing relevant exists. |
| `forget` | Permanently delete a memory. |

Set `COGMEM_TOOLS=full` for four more: `search_memory` (filters by space, type and validity),
`get_context` (a prompt-ready block within a token budget), `update_memory` (edits with
version history) and `get_memory`.

Behind the tools is a small, well-tested engine: duplicate merging, supersession, validity
windows, version history and hybrid ranking. See **[docs/architecture.md](docs/architecture.md)**.

## Manage memory from the terminal

```bash
cogmem recall "how do we deploy?"     # search project + personal memory
cogmem remember "Staging is at staging.example.com" --scope project
cogmem list                           # newest first
cogmem forget <id>                    # permanent delete
cogmem export -o memories.jsonl       # everything you've stored, as JSON Lines
cogmem doctor                         # diagnose the install
```

## Better semantic recall

The default embedder matches on words and word fragments. For recall by meaning ("how do we
ship?" → "deploys go through GitHub Actions"), install the local neural model. It runs on CPU
and needs no API key:

```bash
pipx install "cogmem-mcp[local-embeddings]"
cogmem reindex
```

## Use it from Python

```python
from cogmem import MemoryService

memory = MemoryService.from_config()
memory.remember("The API uses FastAPI and PostgreSQL", space_id="project:shop")
print(memory.get_context("add a new endpoint", space_ids=["project:shop"]).text)
```

## Privacy and security

Everything stays in `~/.cogmem/cogmem.db` on your machine. CogMem refuses writes that look like
API keys, tokens or private keys. `forget` overwrites deleted data on disk, and recalled memories
are marked as untrusted reference data so agents don't follow instructions stored inside them.
See [SECURITY.md](SECURITY.md) to report issues.

## Roadmap

- [x] Core engine: hybrid retrieval, dedup, supersession, validity windows, history
- [x] MCP server and CLI, with automatic project scoping
- [ ] PostgreSQL + pgvector backend with multi-tenant isolation
- [ ] Retrieval benchmark suite in CI (recall@k, MRR)
- [ ] Hosted remote MCP with one-click OAuth connectors
- [ ] Web memory viewer (browse, edit, delete, export)
- [ ] Entity graph, contradiction detection and memory consolidation

Details: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## Contributing

Issues and PRs are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) to get set up; the whole
suite runs in about 15 seconds. If CogMem saves you from re-explaining your codebase, a ⭐
helps others find it.

## License

[MIT](LICENSE)
