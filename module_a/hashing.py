"""File hashes and a perceptual difference hash (dHash)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dhash(img: Image.Image, hash_size: int = 8) -> str:
    """64-bit difference hash as 16 hex chars. Robust to mild brightness/noise, NOT to rotation/crop."""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = 0
    for flag in (a[:, :-1] > a[:, 1:]).flatten():
        bits = (bits << 1) | int(flag)
    return f"{bits:0{hash_size * hash_size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()
