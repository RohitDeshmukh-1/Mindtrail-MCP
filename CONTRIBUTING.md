# Contributing to Mindtrail

Thanks for helping. Small, focused pull requests get reviewed fastest.

## Setup

```bash
git clone https://github.com/RohitDeshmukh-1/Mindtrail-MCP && cd Mindtrail-MCP
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                 # optional: runs ruff on every commit
```

## Before you open a PR

```bash
ruff format src tests
ruff check src tests
mypy src
pytest
```

CI runs the same checks on Linux, macOS and Windows with Python 3.11–3.13.

## Ground rules

- **Business logic lives in `MemoryService`.** MCP tools, the CLI and (soon) the REST API are
  thin adapters. Don't duplicate rules across transports.
- **Every new feature and every security boundary needs a test.** Tenant and space isolation
  tests must never be weakened.
- **No paid API in the core path.** Basic storage and recall must work offline.
- **No new infrastructure without a measured need.** Prefer the standard library and existing
  dependencies.
- **No unverified claims.** Don't document a client integration as supported until someone has
  tested it, and don't state performance numbers without a reproducible benchmark.

## Where to start

Issues labelled `good first issue` are scoped to be finishable in an afternoon. The roadmap is in the
[README](README.md#roadmap).

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
`test:`, `refactor:`, `chore:`.

By contributing you agree your work is released under the [MIT License](LICENSE) and that you
will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
