"""Optional cross-encoder rerankers: score (query, memory) pairs jointly.

A cross-encoder reads the query and the memory together instead of comparing two vectors. It
only rescores the candidates that hybrid retrieval returns.

Reranking is **off by default**. On the bundled benchmark, every small reranker available
through fastembed ranked developer memories *worse* than the bge embeddings alone. The MS MARCO
models reward word overlap over meaning on short, technical text (see benchmarks/README.md).
The stage is kept so that better rerankers can be measured and enabled with
``MINDTRAIL_RERANKER=<model id>``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

# Best score floor found per model in the dev-set screen (on top of bge-small); used only
# when a reranker is explicitly enabled.
_MIN_SCORES: dict[str, float] = {
    "Xenova/ms-marco-MiniLM-L-6-v2": -10.0,
    "Xenova/ms-marco-MiniLM-L-12-v2": -10.0,
    "jinaai/jina-reranker-v1-tiny-en": 0.0,
    "jinaai/jina-reranker-v1-turbo-en": -4.0,
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
    """``auto`` and ``none`` mean no reranker (none has beaten the embeddings yet); any other
    value is a fastembed cross-encoder model id to enable explicitly."""
    if kind in ("auto", "none"):
        return None
    return FastEmbedReranker(kind, cache_dir=cache_dir)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class FastEmbedReranker:
    def __init__(
        self,
        model: str,
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
