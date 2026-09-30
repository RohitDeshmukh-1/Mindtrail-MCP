"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from mindtrail.core.models import DEFAULT_SPACE, validate_space_id

EmbedderKind = Literal["auto", "hashing", "fastembed"]
_EMBEDDERS: tuple[EmbedderKind, ...] = ("auto", "hashing", "fastembed")


@dataclass(frozen=True)
class MindtrailConfig:
    home: Path
    embedder: EmbedderKind = "auto"
    default_space: str = DEFAULT_SPACE
    reranker: str = "auto"  # "auto", "none" or a fastembed cross-encoder model id
    embedding_model: str | None = None  # fastembed model id; None = the calibrated default

    def __post_init__(self) -> None:
        if self.embedder not in _EMBEDDERS:
            raise ValueError(f"embedder must be one of {_EMBEDDERS}, got {self.embedder!r}")
        object.__setattr__(self, "default_space", validate_space_id(self.default_space))

    @property
    def db_path(self) -> Path:
        return self.home / "mindtrail.db"

    @property
    def model_dir(self) -> Path:
        return self.home / "models"

    @classmethod
    def from_env(cls) -> MindtrailConfig:
        home = Path(os.environ.get("MINDTRAIL_HOME") or Path.home() / ".mindtrail").expanduser()
        return cls(
            home=home,
            embedder=cast(EmbedderKind, os.environ.get("MINDTRAIL_EMBEDDER", "auto")),
            default_space=os.environ.get("MINDTRAIL_DEFAULT_SPACE", DEFAULT_SPACE),
            reranker=os.environ.get("MINDTRAIL_RERANKER", "auto").strip() or "auto",
            embedding_model=os.environ.get("MINDTRAIL_EMBEDDING_MODEL", "").strip() or None,
        )
