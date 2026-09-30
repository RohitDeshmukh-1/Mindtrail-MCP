# Connecting Mindtrail to your agent

Mindtrail runs as a local MCP server over stdio. Install it once and every MCP client on your
machine can share the same memory.

```bash
pipx install mindtrail         # or: uv tool install mindtrail
mindtrail doctor               # check the install
```

Memories are scoped automatically: facts about the current repository go to a **project**
space (detected from the git remote, so every tool in the same repo shares it) and facts about
you go to your **personal** space. `recall` searches both.

| Client | Status |
|---|---|
| Claude Code | ✅ Verified: the server connects and passes the health check |
| Custom Python clients (MCP SDK) | ✅ Verified by the automated test suite |
| Cursor, VS Code, Codex | ⚠️ Configuration follows each client's documented MCP format but has not yet been verified. Reports welcome |

## Claude Code

```bash
claude mcp add mindtrail --scope user -- mindtrail serve
```

`--scope user` makes Mindtrail available in every project. Check it with `claude mcp list`.

## Cursor

Add to `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project):

```json
{
  "mcpServers": {
    "mindtrail": { "command": "mindtrail", "args": ["serve"] }
  }
}
```

## VS Code (agent mode)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "mindtrail": { "type": "stdio", "command": "mindtrail", "args": ["serve"] }
  }
}
```

## Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.mindtrail]
command = "mindtrail"
args = ["serve"]
```

## Custom Python client

```python
from mcp import Client, StdioServerParameters

async with Client(StdioServerParameters(command="mindtrail", args=["serve"])) as client:
    await client.call_tool("remember", {"content": "Deploys go through GitHub Actions"})
    result = await client.call_tool("recall", {"query": "how do we deploy?"})
```

## Try it

1. In one session: *"Remember that this project uses conventional commits."*
2. Start a **new** session (or switch tools) and ask: *"How should I write commit messages here?"*
3. From a terminal: `mindtrail list` shows everything stored.

## Options

Set these as environment variables in the client's server config (for example
`claude mcp add mindtrail -e MINDTRAIL_TOOLS=full -- mindtrail serve`).

| Variable | Default | Meaning |
|---|---|---|
| `MINDTRAIL_TOOLS` | `core` | `core` = remember/recall/forget; `full` adds search_memory, get_context, update_memory, get_memory |
| `MINDTRAIL_HOME` | `~/.mindtrail` | Where the database lives |
| `MINDTRAIL_EMBEDDER` | `auto` | `auto`, `hashing` or `fastembed` |
| `MINDTRAIL_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | fastembed model id; `BAAI/bge-small-en-v1.5` is the smaller calibrated option |
| `MINDTRAIL_RERANKER` | `auto` (off) | A fastembed cross-encoder id to enable reranking; none has beaten the embeddings yet |
| `MINDTRAIL_PROJECT` | detected from git | Force a project name |
| `MINDTRAIL_LOG_LEVEL` | `WARNING` | Logs go to stderr |

## Troubleshooting

- **`mindtrail: command not found`**: the client can't see your PATH. Use the full path from
  `which mindtrail` (macOS/Linux) or `where mindtrail` (Windows) as the command.
- **Memories from one repo show up in another**: both folders share a git remote, which is by
  design. Set `MINDTRAIL_PROJECT` to separate them.
- **Anything else**: run `mindtrail doctor` and include its output in a bug report.
