from pathlib import Path

import pytest

from mindtrail.core.project import detect_project, normalize_remote


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINDTRAIL_PROJECT", raising=False)


def _repo(root: Path, remote: str | None = None) -> Path:
    git = root / ".git"
    git.mkdir(parents=True)
    config = "[core]\n\tbare = false\n"
    if remote:
        config += '[remote "upstream"]\n\turl = https://example.com/other.git\n'
        config += f'[remote "origin"]\n\turl = {remote}\n\tfetch = +refs/heads/*\n'
    (git / "config").write_text(config)
    return root


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:Owner/Repo.git",
        "https://github.com/owner/repo.git",
        "https://user:ghp_secret@github.com/owner/repo",
        "ssh://git@github.com:22/owner/repo.git",
    ],
)
def test_remote_spellings_normalize_identically(url: str) -> None:
    assert normalize_remote(url) == "github.com/owner/repo"


def test_same_remote_gives_same_space_across_checkouts(tmp_path: Path) -> None:
    a = _repo(tmp_path / "a" / "repo", "git@github.com:owner/repo.git")
    b = _repo(tmp_path / "b" / "repo", "https://github.com/owner/repo")
    pa, pb = detect_project(a), detect_project(b)
    assert pa and pb and pa.space_id == pb.space_id
    assert pa.space_id.startswith("project:repo-")


def test_detects_from_subdirectory(tmp_path: Path) -> None:
    root = _repo(tmp_path / "My Project")
    nested = root / "src" / "pkg"
    nested.mkdir(parents=True)
    project = detect_project(nested)
    assert project and project.root == root.resolve() and project.name == "my-project"


def test_no_repo_means_no_project(tmp_path: Path) -> None:
    assert detect_project(tmp_path) is None


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINDTRAIL_PROJECT", "Acme Web")
    project = detect_project(tmp_path)
    assert project and project.space_id == "project:acme-web"
