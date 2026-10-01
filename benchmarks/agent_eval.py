"""Agent-in-the-loop evaluation: does a real coding agent use Mindtrail across sessions?

For each fact, a "teach" session mentions it in passing (the agent is never told to use
Mindtrail), then a separate "ask" session gives a task that depends on the fact, worded
differently. A control runs the same ask sessions without Mindtrail. Every session is a
separate ``claude -p`` process, so nothing carries over except what Mindtrail stored.

Requires the Claude Code CLI and a Mindtrail install with the [semantic] extra. Each fact set
costs 12 agent sessions (18 with --control). Run from the repository root:

    python benchmarks/agent_eval.py --facts 1 --control
    python benchmarks/agent_eval.py --facts 2 --mindtrail /path/to/mindtrail --out results.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Fact:
    id: str
    teach: str
    ask: str
    correct: Callable[[str], bool]  # does the answer apply the fact (not just mention it)?


def _code(text: str) -> str:
    match = re.search(r"```[^\n]*\n(.*?)```", text, re.S)
    return match.group(1) if match else text


def _first_line(text: str) -> str:
    return next((line.strip(" `*>-") for line in _code(text).splitlines() if line.strip()), "")


def _recommends_epoch_ms(text: str) -> bool:
    # Epoch ms must come before any ISO 8601 recommendation; an answer that picks ISO and only
    # lists epoch ms as a fallback did not apply the fact.
    epoch = re.search(
        r"epoch millisecond|millisecond.{0,40}epoch|epoch.{0,40}millisecond", text, re.I
    )
    iso = re.search(r"ISO[ -]?8601|RFC ?3339", text, re.I)
    return bool(epoch) and (iso is None or epoch.start() < iso.start())


FACT_SETS: dict[str, list[Fact]] = {
    "1": [
        Fact(
            "release",
            "Heads up for this repo: we release by pushing a git tag like v1.2.3, and CI "
            "publishes from the tag. Nobody runs the publish script by hand.",
            "I just finished a feature and want to cut a release of this project. What do I do?",
            lambda a: bool(
                re.search(
                    r"publish(es|ing)? from (that|the) tag|CI (then )?publish|"
                    r"by hand|publish script",
                    a,
                    re.I,
                )
            ),
        ),
        Fact(
            "pkg",
            "FYI, this project uses pnpm. Please never use npm or yarn here.",
            "How do I add lodash as a dependency to this project? Just give me the command.",
            lambda a: _first_line(a).lower().startswith("pnpm"),
        ),
        Fact(
            "db-port",
            "Note: our staging Postgres listens on port 5433, not the default 5432.",
            "What connection string should I use for the staging database? "
            "Host is staging.internal, db is app.",
            lambda a: "5433" in a,
        ),
        Fact(
            "commits",
            "We write commit messages in Conventional Commits format here "
            "(feat:, fix:, chore: and so on).",
            "Write a one-line commit message for a change that fixes the login redirect loop.",
            lambda a: bool(re.match(r"fix(\([^)]*\))?:", _first_line(a), re.I)),
        ),
        Fact(
            "owner",
            "Just so you know, src/payments is owned by the payments team and every change "
            "there needs a review from @payments-team.",
            "I want to refactor src/payments/charge.ts next week. "
            "Anything I should know before I start?",
            lambda a: "payments-team" in a,
        ),
        Fact(
            "flags",
            "Feature flags live in config/flags.yaml. Never hardcode a flag in code.",
            "Where should I add a new feature flag for dark mode?",
            lambda a: bool(re.search(r"flags\.ya?ml", a)),
        ),
    ],
    "2": [
        Fact(
            "test-cmd",
            "Our test command is `make check`; it runs lint and tests together. "
            "Don't call pytest directly.",
            "How do I run the tests before I open a PR?",
            lambda a: "make check" in a,
        ),
        Fact(
            "deploy-branch",
            "We deploy to production from the `release` branch, not from main.",
            "My change was just merged to main. When will it reach production?",
            lambda a: bool(re.search(r"`?release`? branch", a, re.I)),
        ),
        Fact(
            "py39",
            "We still support Python 3.9, so don't use the `X | Y` union syntax in type hints.",
            "Write the type hint for a function parameter `name` that accepts a str or None.",
            lambda a: bool(re.search(r"Optional\[str\]|Union\[str, None\]", _code(a))),
        ),
        Fact(
            "logging",
            "All logging goes through the `log_event()` helper in utils/telemetry.py, "
            "never print() or the logging module directly.",
            "Add a debug line that logs the user id when checkout starts. "
            "What code should I write?",
            lambda a: "log_event" in a,
        ),
        Fact(
            "timestamps",
            "Timestamps in our API responses are always Unix epoch milliseconds, "
            "never ISO strings.",
            "What format should the created_at field use in the response of the new "
            "/orders endpoint?",
            _recommends_epoch_ms,
        ),
        Fact(
            "oncall",
            "The on-call channel for the checkout service is #shop-oncall on Slack.",
            "The checkout service is throwing 500s in production. Who should I alert?",
            lambda a: "shop-oncall" in a,
        ),
    ],
}


def _session(prompt: str, repo: Path, model: str, mcp_config: Path | None) -> dict[str, Any]:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--setting-sources",
        "local",
        "--tools",
        "Read,Glob,Grep",
        "--disallowedTools",
        "Write,Edit,Bash",
        "--strict-mcp-config",
    ]
    if mcp_config:
        cmd += ["--mcp-config", str(mcp_config), "--allowedTools", "mcp__mindtrail"]
    out = subprocess.run(
        cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=600
    )
    tools: list[str] = []
    answer = ""
    for line in out.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event["message"].get("content", []):
                if block.get("type") == "tool_use":
                    tools.append(block["name"].removeprefix("mcp__mindtrail__"))
        elif event.get("type") == "result":
            answer = event.get("result", "")
            if event.get("is_error"):
                sys.exit(f"agent session failed (usage limit?): {answer}")
    if not answer:
        sys.exit(f"agent session produced no answer: {out.stderr[-500:]}")
    return {"tools": tools, "answer": answer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--facts", choices=sorted(FACT_SETS), default="1")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument(
        "--mindtrail",
        default=shutil.which("mindtrail"),
        help="mindtrail executable to test (default: the one on PATH)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path.home() / ".mindtrail" / "models",
        help="reuse downloaded embedding models from here",
    )
    parser.add_argument("--control", action="store_true", help="also run without Mindtrail")
    parser.add_argument("--out", type=Path, help="write per-session results as JSON")
    args = parser.parse_args()
    if not args.mindtrail:
        sys.exit("mindtrail not found; pass --mindtrail")

    with tempfile.TemporaryDirectory(prefix="mindtrail-agent-eval-") as tmp:
        work = Path(tmp)
        repo, home = work / "acme-shop", work / "home"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/example/acme-shop.git"],
            cwd=repo,
            check=True,
        )
        (repo / "README.md").write_text("# acme-shop\n")
        if args.models_dir.is_dir():
            shutil.copytree(args.models_dir, home / "models")
        config = work / "mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "mindtrail": {
                            "command": args.mindtrail,
                            "args": ["serve"],
                            "env": {"MINDTRAIL_HOME": str(home)},
                        }
                    }
                }
            )
        )

        facts = FACT_SETS[args.facts]
        rows: list[dict[str, Any]] = []
        for fact in facts:
            result = _session(fact.teach, repo, args.model, config)
            rows.append({"fact": fact.id, "phase": "teach", **result})
            print(f"teach  {fact.id:14s} remember={'remember' in result['tools']}", flush=True)
        conditions = [("mindtrail", config)] + ([("control", None)] if args.control else [])
        for fact in facts:
            for name, mcp in conditions:
                result = _session(fact.ask, repo, args.model, mcp)
                ok = fact.correct(str(result["answer"]))
                rows.append({"fact": fact.id, "phase": f"ask-{name}", "correct": ok, **result})
                print(
                    f"ask    {fact.id:14s} {name:9s} correct={ok} "
                    f"recall={'recall' in result['tools']}",
                    flush=True,
                )

    teach = [r for r in rows if r["phase"] == "teach"]
    print(
        f"\nremember called unprompted: {sum('remember' in r['tools'] for r in teach)}/{len(teach)}"
    )
    for name, _ in conditions:
        asks = [r for r in rows if r["phase"] == f"ask-{name}"]
        print(
            f"{name:9s} correct {sum(bool(r['correct']) for r in asks)}/{len(asks)}, "
            f"recall called {sum('recall' in r['tools'] for r in asks)}/{len(asks)}"
        )
    if args.out:
        args.out.write_text(
            json.dumps({"model": args.model, "facts": args.facts, "rows": rows}, indent=2)
        )


if __name__ == "__main__":
    main()
