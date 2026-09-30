"""Pluggable embedding providers."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from mindtrail.core.config import EmbedderKind
from mindtrail.embeddings.base import EmbeddingProvider
from mindtrail.embeddings.hashing import HashingEmbedder

logger = logging.getLogger(__name__)

__all__ = ["EmbeddingProvider", "HashingEmbedder", "create_embedder"]


def create_embedder(
    kind: EmbedderKind = "auto", *, model: str | None = None, cache_dir: Path | None = None
) -> EmbeddingProvider:
    """Build an embedder. ``auto`` uses fastembed when installed, else the offline hashing model.

    ``model`` picks the fastembed model (default: bge-base-en-v1.5). ``cache_dir`` is where
    neural model files are stored (default: fastembed's cache).
    """
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed" or importlib.util.find_spec("fastembed") is not None:
        from mindtrail.embeddings.fastembed_provider import DEFAULT_MODEL, FastEmbedProvider

        return FastEmbedProvider(model or DEFAULT_MODEL, cache_dir=cache_dir)
    logger.info("fastembed not installed; using offline hashing embedder")
    return HashingEmbedder()
