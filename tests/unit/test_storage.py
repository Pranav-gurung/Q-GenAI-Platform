import pytest

from backend.core.exceptions import StorageError
from backend.services.storage import LocalImageStorage


def test_save_load_delete(tmp_path):
    storage = LocalImageStorage(tmp_path)
    obj = storage.save("INS-1/IMG-1.png", b"data", "image/png")
    assert obj.backend == "local" and obj.size_bytes == 4
    assert storage.load(obj.reference) == b"data"
    storage.delete(obj.reference)
    with pytest.raises(StorageError):
        storage.load(obj.reference)
    assert storage.healthy()


def test_path_traversal_rejected(tmp_path):
    storage = LocalImageStorage(tmp_path / "images")
    with pytest.raises(StorageError):
        storage.save("../escape.png", b"x", "image/png")
    with pytest.raises(StorageError):
        storage.load("/etc/passwd")
