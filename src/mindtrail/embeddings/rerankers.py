"""Cross-encoder rerankers: score (query, memory) pairs jointly for precise relevance.

A bi-encoder embedding compares two independently computed vectors. A cross-encoder reads the
query and the memory together, which is far better at paraphrase and at telling "related"
from "actually answers the question". It is too slow to run over a whole store, so it only
rescores the few candidates that hybrid retrieval returns.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"

# Score floor for each model: a candidate below it is not relevant, even as the best available
# match. Calibrated on the bundled dev benchmark (see benchmarks/README.md).
_MIN_SCORES: dict[str, float] = {
    "Xenova/ms-marco-MiniLM-L-6-v2": 0.0,
    "Xenova/ms-marco-MiniLM-L-12-v2": 0.0,
    "jinaai/jina-reranker-v1-tiny-en": 0.0,
    "jinaai/jina-reranker-v1-turbo-en": 0.0,
    "BAAI/bge-reranker-base": 0.0,
}


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def min_score(self) -> float:
        """Raw score (logit) below which a candidate is treated as irrelevant."""
        ...

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def create_reranker(kind: str = "auto", *, cache_dir: Path | None = None) -> Reranker | None:
    """``auto`` enables the default reranker when fastembed is installed; ``none`` disables
    reranking; any other value is a fastembed cross-encoder model id."""
    import importlib.util

    if kind == "none":
        return None
    if kind == "auto":
        if importlib.util.find_spec("fastembed") is None:
            return None
        kind = DEFAULT_RERANKER
    return FastEmbedReranker(kind, cache_dir=cache_dir)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class FastEmbedReranker:
    def __init__(
        self,
        model: str = DEFAULT_RERANKER,
        *,
        min_score: float | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._model_id = model
        self._min_score = _MIN_SCORES.get(model, 0.0) if min_score is None else min_score
        self._cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return f"fastembed-rerank:{self._model_id}"

    @property
    def min_score(self) -> float:
        return self._min_score

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [float(s) for s in self._load().rerank(query, list(documents))]

    def warm_up(self) -> None:
        self._load()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                cache = str(self._cache_dir) if self._cache_dir else None
                self._model = TextCrossEncoder(self._model_id, cache_dir=cache)
            return self._model
