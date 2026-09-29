# Security Policy

Mindtrail stores information that AI agents will later read and act on, so we treat memory
isolation and memory poisoning as security issues.

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through
[GitHub security advisories](https://github.com/OWNER/mindtrail/security/advisories/new).
We aim to acknowledge reports within 3 business days.

Especially in scope:

- Reading or modifying memories across tenants, projects or spaces.
- Stored content that escapes the untrusted-data boundary in `get_context` / `recall` output.
- Secrets that bypass the write-path secret filter in common, realistic formats.
- Deleted memories that remain recoverable from the local database.

## Supported versions

Mindtrail is pre-1.0. Security fixes land on the latest release only.

## How Mindtrail protects your data today

- Local mode keeps everything in one SQLite file under `~/.mindtrail`. Nothing leaves your machine.
- Writes that look like credentials (cloud keys, tokens, private keys) are refused.
- `forget` deletes permanently, and SQLite `secure_delete` overwrites the freed pages.
- Recalled memories are labelled as untrusted reference data, so agents should not follow
  instructions found inside them.
- `mindtrail export` gives you every memory as JSON Lines.
