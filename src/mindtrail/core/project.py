"""Detect the project an agent is working in, so memories can be scoped to it automatically.

The project space id is derived from the git remote when there is one, so Claude Code, Cursor
and Codex working in the same repository share one space. Without a remote, the repository
path is used instead.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

_REMOTE_SECTION = re.compile(r'^\s*\[remote\s+"origin"\]\s*$')
_SECTION = re.compile(r"^\s*\[")
_URL = re.compile(r"^\s*url\s*=\s*(\S+)\s*$")
_SLUG = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class Project:
    name: str
    space_id: str
    root: Path | None


def _slug(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-.") or "project"


def normalize_remote(url: str) -> str:
    """Canonical form of a git remote: ssh and https spellings match, credentials are removed."""
    url = url.strip()
    scp = re.match(r"^[\w.-]+@([\w.-]+):(.+)$", url)  # git@github.com:owner/repo.git
    if scp:
        host, path = scp.groups()
    else:
        url = re.sub(r"^[a-z+]+://", "", url, flags=re.IGNORECASE)
        url = url.split("@", 1)[-1]  # drop user:token@
        host, _, path = url.partition("/")
        host = host.split(":", 1)[0]  # drop port
    path = re.sub(r"\.git/?$", "", path.strip("/"))
    return f"{host}/{path}".lower()


def _origin_url(git_dir: Path) -> str | None:
    try:
        lines = (git_dir / "config").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    in_origin = False
    for line in lines:
        if _SECTION.match(line):
            in_origin = bool(_REMOTE_SECTION.match(line))
        elif in_origin and (match := _URL.match(line)):
            return match.group(1)
    return None


def find_repo_root(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return None


def detect_project(start: Path | None = None) -> Project | None:
    """The current project, from ``MINDTRAIL_PROJECT`` or the enclosing git repository."""
    if override := os.environ.get("MINDTRAIL_PROJECT", "").strip():
        name = _slug(override)
        return Project(name=name, space_id=f"project:{name}", root=None)

    root = find_repo_root((start or Path.cwd()).resolve())
    if root is None:
        return None
    git_path = root / ".git"
    remote = _origin_url(git_path) if git_path.is_dir() else None
    identity = normalize_remote(remote) if remote else str(root).lower()
    # With a remote, the name comes from it too, so clones in differently named folders match.
    name = _slug(identity.rsplit("/", 1)[-1] if remote else root.name)[:80]
    digest = hashlib.sha256(identity.encode()).hexdigest()[:8]
    return Project(name=name, space_id=f"project:{name}-{digest}", root=root)
