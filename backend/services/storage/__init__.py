"""Image storage abstraction."""
from backend.services.storage.base import ImageStorage, StoredObject
from backend.services.storage.factory import build_storage
from backend.services.storage.local import LocalImageStorage

__all__ = ["ImageStorage", "StoredObject", "LocalImageStorage", "build_storage"]
