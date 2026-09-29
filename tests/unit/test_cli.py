import json
from pathlib import Path

import pytest

from cogmem.cli.main import main


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("COGMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COGMEM_EMBEDDER", "hashing")
    monkeypatch.setenv("COGMEM_PROJECT", "cli-test")
    return tmp_path


def test_remember_recall_forget_roundtrip(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["remember", "Build with `make release`"]) == 0
    assert "[project:cli-test]" in capsys.readouterr().out

    assert main(["recall", "how to build a release", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["memory"]["content"] == "Build with `make release`"

    assert main(["forget", hits[0]["memory"]["id"]]) == 0
    capsys.readouterr()
    assert main(["recall", "how to build a release"]) == 0
    assert "No relevant memories." in capsys.readouterr().out


def test_personal_scope_and_list(capsys: pytest.CaptureFixture[str]) -> None:
    main(["remember", "Prefers British spelling", "--scope", "personal"])
    capsys.readouterr()
    assert main(["list", "--space", "personal", "--json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert [r["content"] for r in records] == ["Prefers British spelling"]


def test_export_writes_jsonl(tmp_path: Path) -> None:
    main(["remember", "Fact one"])
    main(["remember", "Fact two"])
    out = tmp_path / "export.jsonl"
    assert main(["export", "-o", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["content"] for line in lines) == ["Fact one", "Fact two"]


def test_errors_exit_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["remember", "token ghp_" + "a" * 36]) == 1
    assert "secrets" in capsys.readouterr().err
    assert main(["forget", "not-a-uuid"]) == 1


def test_doctor_and_init_succeed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0
    assert "ERR" not in capsys.readouterr().out
    assert main(["init"]) == 0
    out = capsys.readouterr().out
    assert "claude mcp add cogmem" in out and "mcpServers" in out
