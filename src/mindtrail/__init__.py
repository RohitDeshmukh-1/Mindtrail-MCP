"""Mindtrail: persistent, searchable memory for AI agents."""

from mindtrail.core.config import MindtrailConfig
from mindtrail.core.models import MemoryContext, MemoryRecord, MemoryType, RememberResult, SearchHit
from mindtrail.memory.service import MemoryService

__version__ = "0.1.0"

__all__ = [
    "MemoryContext",
    "MemoryRecord",
    "MemoryService",
    "MemoryType",
    "MindtrailConfig",
    "RememberResult",
    "SearchHit",
]
