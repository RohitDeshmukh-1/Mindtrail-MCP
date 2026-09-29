"""Local neural embeddings via fastembed (ONNX, CPU, no API key). Optional dependency."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mindtrail.embeddings.base import Matrix, Vector

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        # Calibrated on the bundled retrieval benchmark (see benchmarks/README.md).
        min_similarity: float = 0.65,
        keyword_support_similarity: float = 0.60,
        cache_dir: Path | None = None,
    ) -> None:
        self._model_id = model
        self._cache_dir = cache_dir
        self._min_similarity = min_similarity
        self._keyword_support = keyword_support_similarity
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return f"fastembed:{self._model_id}"

    @property
    def min_similarity(self) -> float:
        return self._min_similarity

    @property
    def keyword_support_similarity(self) -> float:
        return self._keyword_support

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.stack(list(self._load().passage_embed(list(texts)))))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = _normalize(np.stack(list(self._load().query_embed(text))))[0]
        return vector

    def warm_up(self) -> None:
        """Load the model now (downloading it on first use) instead of on the first query."""
        self._load()

    def _load(self) -> Any:
        # Loaded lazily: the first call downloads the model (~65 MB) into the fastembed cache.
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                cache = str(self._cache_dir) if self._cache_dir else None
                self._model = TextEmbedding(self._model_id, cache_dir=cache)
            return self._model


def _normalize(matrix: Any) -> Matrix:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    result: Matrix = matrix / np.where(norms == 0, 1.0, norms)
    return result
