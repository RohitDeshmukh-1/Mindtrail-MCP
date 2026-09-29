"""Errors raised by the memory engine. Transports map these to their own error formats."""

from uuid import UUID


class MindtrailError(Exception):
    """Base class for all Mindtrail errors."""


class InvalidMemoryError(MindtrailError):
    """Input failed validation."""


class MemoryNotFoundError(MindtrailError):
    def __init__(self, memory_id: UUID | str) -> None:
        super().__init__(f"memory {memory_id} not found")
        self.memory_id = str(memory_id)


class SecretDetectedError(MindtrailError):
    """Content looks like it contains a credential; storing it is refused."""

    def __init__(self, kinds: list[str]) -> None:
        super().__init__(
            "refusing to store content that appears to contain secrets: " + ", ".join(kinds)
        )
        self.kinds = kinds
