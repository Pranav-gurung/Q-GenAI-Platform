from __future__ import annotations

from backend.core.config import Settings
from backend.services.storage.base import ImageStorage
from backend.services.storage.local import LocalImageStorage


def build_storage(settings: Settings) -> ImageStorage:
    if settings.storage_backend == "local":
        return LocalImageStorage(settings.storage_path)
    # "s3" / "azure" / "gcs": add implementations here when needed.
    raise ValueError(f"STORAGE_BACKEND={settings.storage_backend!r} is not supported (available: local)")
