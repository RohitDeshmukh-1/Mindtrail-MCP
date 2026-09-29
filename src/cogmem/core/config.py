"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from cogmem.core.models import DEFAULT_SPACE, validate_space_id

EmbedderKind = Literal["auto", "hashing", "fastembed"]
_EMBEDDERS: tuple[EmbedderKind, ...] = ("auto", "hashing", "fastembed")


@dataclass(frozen=True)
class CogMemConfig:
    home: Path
    embedder: EmbedderKind = "auto"
    default_space: str = DEFAULT_SPACE

    def __post_init__(self) -> None:
        if self.embedder not in _EMBEDDERS:
            raise ValueError(f"embedder must be one of {_EMBEDDERS}, got {self.embedder!r}")
        object.__setattr__(self, "default_space", validate_space_id(self.default_space))

    @property
    def db_path(self) -> Path:
        return self.home / "cogmem.db"

    @classmethod
    def from_env(cls) -> CogMemConfig:
        home = Path(os.environ.get("COGMEM_HOME") or Path.home() / ".cogmem").expanduser()
        return cls(
            home=home,
            embedder=cast(EmbedderKind, os.environ.get("COGMEM_EMBEDDER", "auto")),
            default_space=os.environ.get("COGMEM_DEFAULT_SPACE", DEFAULT_SPACE),
        )
