"""Storage interface. The pipeline only sees opaque `reference` strings, so swapping
local disk for S3 / Azure Blob / GCS means adding one class + one factory entry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredObject:
    backend: str
    reference: str  # opaque, backend-specific (relative path, object key, ...)
    size_bytes: int


class ImageStorage(ABC):
    backend_name: str

    @abstractmethod
    def save(self, key: str, data: bytes, content_type: str) -> StoredObject: ...

    @abstractmethod
    def load(self, reference: str) -> bytes: ...

    @abstractmethod
    def delete(self, reference: str) -> None: ...

    @abstractmethod
    def healthy(self) -> bool: ...
