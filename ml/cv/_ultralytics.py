"""Single import point for Ultralytics, with its global side effects neutralised.

Importing `ultralytics` monkey-patches `PIL.Image.open` so that an unidentifiable file triggers a
runtime `pip install pi-heif` attempt (and, offline, a ModuleNotFoundError). That would change
Part 1 upload validation from a clean CORRUPT_IMAGE (422) into a 500 and must not happen in a
server. HEIF is not an accepted upload type, so we restore Pillow's original `Image.open`.
Environment defaults keep Ultralytics quiet, offline and without auto-installs.
"""
from __future__ import annotations

import os

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")


def import_ultralytics():
    """Return the `ultralytics` module with Pillow's original Image.open restored."""
    import ultralytics
    from PIL import Image

    try:
        from ultralytics.utils import patches
        original = getattr(patches, "_image_open", None)
        if original is not None and Image.open is not original:
            Image.open = original
    except ImportError:  # layout changed in a future version: nothing to undo
        pass
    return ultralytics
