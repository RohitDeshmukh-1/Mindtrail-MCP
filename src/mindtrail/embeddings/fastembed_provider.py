"""Local neural embeddings via fastembed (ONNX, CPU, no API key). Optional dependency."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mindtrail.embeddings.base import Matrix, Vector

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_RETRIEVAL_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelProfile:
    """Per-model prompt format and similarity calibration.

    Thresholds are cosine similarities, calibrated on the bundled dev benchmark
    (see benchmarks/README.md). Similarity scales differ a lot between models, so an
    uncalibrated model falls back to the defaults of the closest known model.
    """

    query_prefix: str = ""
    document_prefix: str = ""
    min_similarity: float = 0.65  # relevant on vector evidence alone
    keyword_support: float = 0.60  # confirms a keyword hit is not incidental
    candidate_similarity: float = 0.45  # loose floor for reranker candidates


PROFILES: dict[str, ModelProfile] = {
    "BAAI/bge-small-en-v1.5": ModelProfile(),
}


class FastEmbedProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        profile: ModelProfile | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._model_id = model
        self._profile = profile or PROFILES.get(model, ModelProfile())
        self._cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        # Stored vectors depend on the document prefix, so it is part of the identity.
        suffix = "+docprefix" if self._profile.document_prefix else ""
        return f"fastembed:{self._model_id}{suffix}"

    @property
    def min_similarity(self) -> float:
        return self._profile.min_similarity

    @property
    def keyword_support_similarity(self) -> float:
        return self._profile.keyword_support

    @property
    def candidate_similarity(self) -> float:
        return self._profile.candidate_similarity

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefix = self._profile.document_prefix
        inputs = [prefix + text for text in texts]
        return _normalize(np.stack(list(self._load().embed(inputs))))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = _normalize(
            np.stack(list(self._load().embed([self._profile.query_prefix + text])))
        )[0]
        return vector

    def warm_up(self) -> None:
        """Load the model now (downloading it on first use) instead of on the first query."""
        self._load()

    def _load(self) -> Any:
        # Loaded lazily: the first call downloads the model into the cache directory.
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
