from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from mindtrail.core.models import MemoryType

Category = Literal["lexical", "paraphrase", "temporal", "isolation", "negative"]
BUNDLED = {
    "dev": "retrieval_v1.json",
    "holdout": "retrieval_holdout_v1.json",
    "holdout-v2": "retrieval_holdout_v2.json",
}


class BenchMemory(BaseModel):
    key: str
    space: str
    content: str
    type: MemoryType = MemoryType.SEMANTIC
    importance: float = 0.5
    age_days: float = Field(default=30.0, ge=0)
    supersedes: str | None = None
    valid_for_days: float | None = Field(default=None, gt=0)


class BenchQuery(BaseModel):
    id: str
    category: Category
    project: str | None
    query: str
    relevant: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)

    @property
    def spaces(self) -> list[str]:
        return [f"project:{self.project}", "personal"] if self.project else ["personal"]


class BenchDataset(BaseModel):
    name: str
    version: str
    description: str
    memories: list[BenchMemory]
    queries: list[BenchQuery]

    @model_validator(mode="after")
    def _check_references(self) -> BenchDataset:
        keys = [m.key for m in self.memories]
        if len(keys) != len(set(keys)):
            raise ValueError("memory keys must be unique")
        ids = [q.id for q in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("query ids must be unique")
        known = set(keys)
        for memory in self.memories:
            if memory.supersedes and memory.supersedes not in known:
                raise ValueError(f"{memory.key} supersedes unknown memory {memory.supersedes}")
        for query in self.queries:
            unknown = set(query.relevant + query.forbidden) - known
            if unknown:
                raise ValueError(f"{query.id} references unknown memories {sorted(unknown)}")
            if set(query.relevant) & set(query.forbidden):
                raise ValueError(f"{query.id} lists a memory as both relevant and forbidden")
            if query.category == "negative" and query.relevant:
                raise ValueError(f"negative query {query.id} must not have relevant memories")
        return self


def load_dataset(source: str | Path = "dev") -> BenchDataset:
    """Load a bundled dataset by name (``dev`` or ``holdout``) or a dataset file by path."""
    if isinstance(source, str) and source in BUNDLED:
        resource = files("mindtrail.evaluation").joinpath("data", BUNDLED[source])
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return BenchDataset.model_validate_json(text)
