<div align="center">

# 🧭 Mindtrail

**Leave a trail your AI agents can follow.**

Persistent, local-first memory for coding agents over MCP. Tell your agent something once,
and every future session remembers it.

[![CI](https://github.com/OWNER/mindtrail/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/mindtrail/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

Every new agent session starts from zero. You explain your conventions again, re-state your
preferences, and re-describe decisions you made last week. **Mindtrail gives your agents a shared,
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

## Does it actually help?

We tested it with real Claude Code sessions. In one session the user mentions a project fact
in passing ("FYI, this project uses pnpm"). A fresh session later gets a task that depends on
it ("how do I add lodash?"). The agent is never told to use Mindtrail.

| | stored the fact on its own | applied it in a later session |
|---|---:|---:|
| Claude Code without Mindtrail | – | 0/12 |
| Claude Code with `mindtrail[semantic]` | **36/36** | **36/36** |

Without memory, the agent answers from habit every time: `npm install lodash`, a commit
message in the wrong format, port 5432 instead of your 5433. With Mindtrail it checks first
and gets your project's answer. This is a small test (12 facts run three times, one model,
an empty repository), not a measure of task success on large codebases.
[Method, raw results and how to reproduce](benchmarks/README.md#agent-evaluation).

## Quickstart

```bash
pipx install "mindtrail[semantic]"                         # or: uv tool install "mindtrail[semantic]"
mindtrail init                                             # downloads the embedding model once (~210 MB)
claude mcp add mindtrail --scope user -- mindtrail serve   # Claude Code
```

Cursor, VS Code, Codex and custom clients are covered in
**[docs/integrations.md](docs/integrations.md)**, and `mindtrail init` prints each config.
Want the smallest install? `pipx install mindtrail` skips the model download and matches on
words instead of meaning (see [Better semantic recall](#better-semantic-recall)).

Then try it:

```text
Session 1 › FYI, this project uses conventional commits and squash merges.
Session 2 › Write a commit message for these changes.
            → the agent recalls the convention and writes "feat(api): add pagination to /orders"
```

## Using it day to day

You don't need special commands. Work as usual and the agent decides what to keep:

- **Mention things once.** "We deploy from the `release` branch", "I prefer pytest over
  unittest", "the flaky test was a timezone bug, fixed by pinning TZ=UTC". The agent stores
  facts like these on its own. Say "remember that…" when you want to be sure.
- **Project vs. personal.** Facts about the repository go to a project space, shared by every
  clone of the same git remote. Facts about you ("I like short answers") go to your personal
  space and apply in every project.
- **Things change.** Say "we moved from npm to pnpm" and the agent replaces the old memory
  instead of keeping both. The old one is kept as history but no longer recalled.
- **Ask what it knows.** "What do you remember about this project?" or, from the terminal,
  `mindtrail list` and `mindtrail recall "<question>"`.
- **Forget anything.** "Forget the staging URL" in chat, or `mindtrail forget <id>`. Deletes
  are permanent.
- **Switch tools freely.** Claude Code, Cursor and Codex pointed at the same Mindtrail share
  one memory, so a convention taught in one tool is known in all of them.

Good things to store: conventions, commands, ownership, where config lives, decisions and why
they were made, root causes of tricky bugs. Don't bother with what the code or git history
already says. Secrets are refused automatically.

## How it works

Your agent gets three tools, and Mindtrail's server instructions tell it when to use them:

| Tool | What it does |
|---|---|
| `remember` | Store one fact, preference, decision or event, scoped to the **project** or **personal** space. Pass `replaces` to supersede an outdated memory. |
| `recall` | Find relevant memories from the current project plus your personal space. Returns nothing when nothing relevant exists. |
| `forget` | Permanently delete a memory. |

Set `MINDTRAIL_TOOLS=full` for four more: `search_memory` (filters by space, type and validity),
`get_context` (a prompt-ready block within a token budget), `update_memory` (edits with
version history) and `get_memory`.

Behind the tools is a small, well-tested engine: duplicate merging, supersession, validity
windows, version history and hybrid ranking. See **[docs/architecture.md](docs/architecture.md)**.

## Manage memory from the terminal

```bash
mindtrail recall "how do we deploy?"   # search project + personal memory
mindtrail remember "Staging is at staging.example.com" --scope project
mindtrail list                         # newest first
mindtrail forget <id>                  # permanent delete
mindtrail export -o memories.jsonl     # everything you've stored, as JSON Lines
mindtrail doctor                       # diagnose the install
```

## Better semantic recall

The default embedder matches on words and word fragments. For recall by meaning ("how do we
deploy?" → "deploys go through GitHub Actions"), install the local neural model. It runs on CPU
and needs no API key:

```bash
pipx install "mindtrail[semantic]"
mindtrail init        # downloads the model once (~210 MB) and re-indexes existing memories
```

For a smaller download (67 MB, somewhat lower recall), set
`MINDTRAIL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`. Memories stored under another model are
re-embedded automatically the next time the server starts.

## Retrieval benchmark

The agent test above is in [Does it actually help?](#does-it-actually-help). On two held-out sets of developer-memory questions (82 in total) that were never used for
tuning ([methodology](benchmarks/README.md)):

| | recall@1 | recall@5 | paraphrase recall@5 | correctly says "nothing stored" | stale/foreign leaks |
|---|---:|---:|---:|---:|---:|
| default install | 58–64% | 68–75% | 44–46% | 82–100% | **0%** |
| `mindtrail[semantic]` | **82–90%** | **93–94%** | **85–89%** | **100%** | **0%** |

This is a small, synthetic retrieval benchmark written by us. Run `mindtrail bench` to
reproduce it, or add your own dataset.

## Use it from Python

```python
from mindtrail import MemoryService

memory = MemoryService.from_config()
memory.remember("The API uses FastAPI and PostgreSQL", space_id="project:shop")
print(memory.get_context("add a new endpoint", space_ids=["project:shop"]).text)
```

## Privacy and security

Everything stays in `~/.mindtrail/mindtrail.db` on your machine. Mindtrail refuses writes that
look like API keys, tokens or private keys. `forget` overwrites deleted data on disk, and recalled memories
are marked as untrusted reference data so agents don't follow instructions stored inside them.
See [SECURITY.md](SECURITY.md) to report issues.

## Roadmap

- [x] Core engine: hybrid retrieval, dedup, supersession, validity windows, history
- [x] MCP server and CLI, with automatic project scoping
- [ ] PostgreSQL + pgvector backend with multi-tenant isolation
- [x] Retrieval benchmark suite in CI (recall@k, MRR, abstention, leak rate)
- [ ] Hosted remote MCP with one-click OAuth connectors
- [ ] Web memory viewer (browse, edit, delete, export)
- [ ] Entity graph, contradiction detection and memory consolidation

## Contributing

Issues and PRs are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) to get set up; the whole
suite runs in about 15 seconds. If Mindtrail saves you from re-explaining your codebase, a ⭐
helps others find it.

## License

[MIT](LICENSE)
