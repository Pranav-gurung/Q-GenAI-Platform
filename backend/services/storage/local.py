"""Local filesystem storage (development / single-node deployments)."""
from __future__ import annotations

import os
from pathlib import Path

from backend.core.exceptions import StorageError
from backend.services.storage.base import ImageStorage, StoredObject


class LocalImageStorage(ImageStorage):
    backend_name = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, reference: str) -> Path:
        path = (self.root / reference).resolve()
        if not path.is_relative_to(self.root):  # keys are server-generated, but never trust paths
            raise StorageError("Invalid storage reference", code="INVALID_STORAGE_REFERENCE")
        return path

    def save(self, key: str, data: bytes, content_type: str) -> StoredObject:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)  # atomic: readers never see a half-written file
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise StorageError("Could not store image") from exc
        return StoredObject(self.backend_name, key, len(data))

    def load(self, reference: str) -> bytes:
        try:
            return self._path(reference).read_bytes()
        except FileNotFoundError as exc:
            raise StorageError("Stored image not found", code="IMAGE_NOT_FOUND") from exc

    def delete(self, reference: str) -> None:
        self._path(reference).unlink(missing_ok=True)

    def healthy(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.W_OK)
