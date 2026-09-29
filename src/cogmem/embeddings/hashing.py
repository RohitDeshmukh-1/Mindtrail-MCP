"""Offline, dependency-free embedder based on feature hashing.

It captures lexical overlap (words plus character trigrams), not deep semantics. It exists so
CogMem works with zero downloads; install the ``local-embeddings`` extra for a neural model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from cogmem.core.text import STOPWORDS, tokenize
from cogmem.embeddings.base import Matrix, Vector

_WORD_WEIGHT = 1.0
_TRIGRAM_WEIGHT = 0.4


class HashingEmbedder:
    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    @property
    def model_name(self) -> str:
        return f"hashing-v1-{self.dimension}"

    @property
    def min_similarity(self) -> float:
        return 0.15

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return np.stack([self._embed(text) for text in texts]) if texts else self._empty()

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _empty(self) -> Matrix:
        return np.zeros((0, self.dimension), dtype=np.float32)

    def _embed(self, text: str) -> Vector:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in tokenize(text):
            if word in STOPWORDS:
                continue
            self._add(vector, "w:" + word, _WORD_WEIGHT)
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                self._add(vector, "t:" + padded[i : i + 3], _TRIGRAM_WEIGHT)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def _add(self, vector: Vector, feature: str, weight: float) -> None:
        # blake2b rather than hash(): stable across processes, so stored vectors stay valid.
        digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        sign = 1.0 if digest >> 63 else -1.0
        vector[digest % self.dimension] += sign * weight
