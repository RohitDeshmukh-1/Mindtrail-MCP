"""Errors raised by the memory engine. Transports map these to their own error formats."""

from uuid import UUID


class CogMemError(Exception):
    """Base class for all CogMem errors."""


class InvalidMemoryError(CogMemError):
    """Input failed validation."""


class MemoryNotFoundError(CogMemError):
    def __init__(self, memory_id: UUID | str) -> None:
        super().__init__(f"memory {memory_id} not found")
        self.memory_id = str(memory_id)


class SecretDetectedError(CogMemError):
    """Content looks like it contains a credential; storing it is refused."""

    def __init__(self, kinds: list[str]) -> None:
        super().__init__(
            "refusing to store content that appears to contain secrets: " + ", ".join(kinds)
        )
        self.kinds = kinds
