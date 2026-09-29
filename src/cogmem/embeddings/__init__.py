"""Pluggable embedding providers."""

from __future__ import annotations

import importlib.util
import logging

from cogmem.core.config import EmbedderKind
from cogmem.embeddings.base import EmbeddingProvider
from cogmem.embeddings.hashing import HashingEmbedder

logger = logging.getLogger(__name__)

__all__ = ["EmbeddingProvider", "HashingEmbedder", "create_embedder"]


def create_embedder(kind: EmbedderKind = "auto") -> EmbeddingProvider:
    """Build an embedder. ``auto`` uses fastembed when installed, else the offline hashing model."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed" or importlib.util.find_spec("fastembed") is not None:
        from cogmem.embeddings.fastembed_provider import FastEmbedProvider

        return FastEmbedProvider()
    logger.info("fastembed not installed; using offline hashing embedder")
    return HashingEmbedder()
