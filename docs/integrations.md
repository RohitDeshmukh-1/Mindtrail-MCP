# Connecting CogMem to your agent

CogMem runs as a local MCP server over stdio. Install it once and every MCP client on your
machine can share the same memory.

```bash
pipx install cogmem-mcp        # or: uv tool install cogmem-mcp
cogmem doctor                  # check the install
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
claude mcp add cogmem --scope user -- cogmem serve
```

`--scope user` makes CogMem available in every project. Check it with `claude mcp list`.

## Cursor

Add to `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project):

```json
{
  "mcpServers": {
    "cogmem": { "command": "cogmem", "args": ["serve"] }
  }
}
```

## VS Code (agent mode)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "cogmem": { "type": "stdio", "command": "cogmem", "args": ["serve"] }
  }
}
```

## Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.cogmem]
command = "cogmem"
args = ["serve"]
```

## Custom Python client

```python
from mcp import Client, StdioServerParameters

async with Client(StdioServerParameters(command="cogmem", args=["serve"])) as client:
    await client.call_tool("remember", {"content": "Deploys go through GitHub Actions"})
    result = await client.call_tool("recall", {"query": "how do we deploy?"})
```

## Try it

1. In one session: *"Remember that this project uses conventional commits."*
2. Start a **new** session (or switch tools) and ask: *"How should I write commit messages here?"*
3. From a terminal: `cogmem list` shows everything stored.

## Options

Set these as environment variables in the client's server config (for example
`claude mcp add cogmem -e COGMEM_TOOLS=full -- cogmem serve`).

| Variable | Default | Meaning |
|---|---|---|
| `COGMEM_TOOLS` | `core` | `core` = remember/recall/forget; `full` adds search_memory, get_context, update_memory, get_memory |
| `COGMEM_HOME` | `~/.cogmem` | Where the database lives |
| `COGMEM_EMBEDDER` | `auto` | `auto`, `hashing` or `fastembed` |
| `COGMEM_PROJECT` | detected from git | Force a project name |
| `COGMEM_LOG_LEVEL` | `WARNING` | Logs go to stderr |

## Troubleshooting

- **`cogmem: command not found`**: the client can't see your PATH. Use the full path from
  `which cogmem` (macOS/Linux) or `where cogmem` (Windows) as the command.
- **Memories from one repo show up in another**: both folders share a git remote, which is by
  design. Set `COGMEM_PROJECT` to separate them.
- **Anything else**: run `cogmem doctor` and include its output in a bug report.
