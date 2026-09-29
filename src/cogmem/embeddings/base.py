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
        """Cosine similarity below which a vector match is treated as irrelevant."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...
