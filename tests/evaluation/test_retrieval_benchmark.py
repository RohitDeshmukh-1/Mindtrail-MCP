"""Retrieval quality gates. Thresholds sit a little below measured results, so a regression
in ranking fails CI while ordinary noise does not. Leak rates must stay exactly zero.

Measured results and methodology: benchmarks/README.md
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mindtrail.cli.main import main
from mindtrail.embeddings import EmbeddingProvider, HashingEmbedder
from mindtrail.evaluation import BenchDataset, load_dataset, run_benchmark
from mindtrail.evaluation.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank

# (dataset, recall@1, recall@5, abstention) minimums
HASHING_GATES = [
    ("dev", 0.60, 0.64, 0.64),
    ("holdout", 0.60, 0.70, 0.75),
    ("holdout-v2", 0.52, 0.62, 0.90),
]
SEMANTIC_GATES = [
    ("dev", 0.78, 0.90, 0.88),
    ("holdout", 0.75, 0.88, 0.90),
    ("holdout-v2", 0.82, 0.88, 0.90),
]


def _check(dataset: str, embedder: EmbeddingProvider, gates: tuple[float, float, float]) -> None:
    report = run_benchmark(load_dataset(dataset), embedder)
    overall = report.overall
    min_r1, min_r5, min_abstention = gates
    assert overall.violations == 0.0, [r.id for r in report.results if r.violated]
    assert overall.recall_at_1 is not None and overall.recall_at_1 >= min_r1
    assert overall.recall_at_k is not None and overall.recall_at_k >= min_r5
    assert overall.abstention is not None and overall.abstention >= min_abstention
    # Queries that share wording with their memory must always be found.
    assert report.by_category["lexical"].recall_at_k == 1.0


@pytest.mark.parametrize(("dataset", "r1", "r5", "abstention"), HASHING_GATES)
def test_hashing_embedder_quality(dataset: str, r1: float, r5: float, abstention: float) -> None:
    _check(dataset, HashingEmbedder(), (r1, r5, abstention))


@pytest.mark.skipif(importlib.util.find_spec("fastembed") is None, reason="needs [semantic]")
@pytest.mark.parametrize(("dataset", "r1", "r5", "abstention"), SEMANTIC_GATES)
def test_semantic_embedder_quality(dataset: str, r1: float, r5: float, abstention: float) -> None:
    from mindtrail.embeddings.fastembed_provider import FastEmbedProvider

    _check(dataset, FastEmbedProvider(), (r1, r5, abstention))


@pytest.mark.parametrize("dataset", ["dev", "holdout", "holdout-v2"])
def test_bundled_datasets_are_valid(dataset: str) -> None:
    data = load_dataset(dataset)
    assert len(data.queries) >= 30
    assert {q.category for q in data.queries} == {
        "lexical", "paraphrase", "temporal", "isolation", "negative",
    }  # fmt: skip


def test_dataset_rejects_dangling_references() -> None:
    raw = {
        "name": "x",
        "version": "1",
        "description": "",
        "memories": [{"key": "a", "space": "personal", "content": "fact"}],
        "queries": [{"id": "q", "category": "lexical", "project": None, "query": "fact",
                     "relevant": ["missing"]}],
    }  # fmt: skip
    with pytest.raises(ValueError, match="unknown memories"):
        BenchDataset.model_validate(raw)


def test_metrics() -> None:
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"b"}) == 0.0
    assert ndcg_at_k(["b", "a"], {"b"}, 5) == 1.0
    assert 0 < ndcg_at_k(["a", "b"], {"b"}, 5) < 1
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)


def test_bench_command_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "report.json"
    assert main(["bench", "--embedder", "hashing", "--dataset", "holdout", "--json", str(out)]) == 0
    assert "| **overall** |" in capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["overall"]["violations"] == 0.0
