"""CogMem: persistent, searchable memory for AI agents."""

from cogmem.core.config import CogMemConfig
from cogmem.core.models import MemoryContext, MemoryRecord, MemoryType, RememberResult, SearchHit
from cogmem.memory.service import MemoryService

__version__ = "0.1.0.dev0"

__all__ = [
    "CogMemConfig",
    "MemoryContext",
    "MemoryRecord",
    "MemoryService",
    "MemoryType",
    "RememberResult",
    "SearchHit",
]
