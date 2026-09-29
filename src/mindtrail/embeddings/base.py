from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]
Matrix = npt.NDArray[np.float32]


class EmbeddingProvider(Protocol):
    """Produces L2-normalized float32 vectors, so dot product equals cosine similarity."""

    @property
    def model_name(self) -> str:
        """Stable identifier stored with each vector; vectors from other models are ignored."""
        ...

    @property
    def min_similarity(self) -> float:
        """Cosine similarity at which a memory is relevant on vector evidence alone."""
        ...

    @property
    def keyword_support_similarity(self) -> float:
        """Minimum cosine similarity that confirms a keyword match is not incidental."""
        ...

    @property
    def candidate_similarity(self) -> float:
        """Loose similarity floor for candidates that a reranker will rescore."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...
