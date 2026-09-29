# Architecture

Mindtrail is a modular monolith. One engine (`MemoryService`) holds all memory logic; each
interface is a thin adapter over it.

```text
 Claude Code · Cursor · Codex · VS Code        your Python code        terminal
            │ MCP (stdio)                            │                     │
            ▼                                        │                     ▼
   mindtrail.mcp.server ──────────┐                  │              mindtrail.cli
                                  ▼                  ▼                     │
                      mindtrail.memory.service.MemoryService  ◀────────────┘
                 ┌──────────────┬────────┴───────┬───────────────┐
                 ▼              ▼                ▼               ▼
            safety.py     retrieval.py      context.py      embeddings/
         (secret filter) (hybrid ranking) (token budget)  (hashing | fastembed)
                                  │
                                  ▼
                      storage/ MemoryRepository
                      └── sqlite.py (FTS5 + vectors)   postgres.py (planned)
```

## Write path

`remember` validates the record, refuses content that looks like a credential, and merges
exact duplicates in the same space. It then embeds the text and stores it. If embedding fails
the memory is still stored: keyword search finds it right away, and `mindtrail reindex` fills in
the vector later. No LLM call is involved.

## Read path

1. **Scope:** the tenant is fixed server-side and the spaces come from the project and personal
   scope. Every SQL query is filtered by both.
2. **Candidates:** an FTS5 BM25 keyword search runs alongside cosine similarity over the stored
   vectors.
3. **Fusion:** reciprocal-rank fusion, then small multiplicative boosts for importance,
   confidence and recency. Because boosts only reorder memories that already matched, they
   cannot promote irrelevant ones.
4. **Filtering:** replaced and expired memories are excluded unless the caller asks for them.
5. **Output:** results carry ids, timestamps and ranking signals. `get_context` wraps them in an
   explicit untrusted-data block that stored content cannot close.

## Why these choices

- **SQLite first:** zero setup, a single file, and good enough for one developer's memory.
  PostgreSQL + pgvector comes next for hosted, multi-tenant use behind the same repository
  interface.
- **Hashing embedder by default:** installs in seconds with no downloads. The neural model is
  one extra away (`mindtrail[semantic]`).
- **Three default tools:** a small, distinct tool surface makes agents call memory tools
  more reliably. Power users can opt into the full set.
- **Project identity from the git remote:** the same repo gets the same memory in every tool
  and every clone, and credentials in remote URLs are stripped before hashing.

The original product specification (written under the working name CogMem) is in
[design/specification.md](design/specification.md).
