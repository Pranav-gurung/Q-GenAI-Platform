"""Test helpers: generate real (tiny) images in memory. No fixture files needed."""
from __future__ import annotations

import io

from PIL import Image


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (64, 48), color=(200, 120, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def corrupt_png() -> bytes:
    """Valid PNG header followed by truncated/garbage data."""
    good = image_bytes("PNG", (256, 256))
    return good[:120] + b"\x00garbage" * 10
