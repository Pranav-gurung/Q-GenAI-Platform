from __future__ import annotations

import numpy as np
from PIL import Image

from backend.pipeline.context import ImageInput
from backend.schemas.common import utcnow
from backend.schemas.image import ImageMetadata


def textured(size: tuple[int, int], seed: int = 0) -> Image.Image:
    """A sharp, mid-brightness image (random blocks + noise), like a normal product photo for the gate."""
    rng = np.random.default_rng(seed)
    w, h = size
    blocks = rng.integers(40, 220, (max(1, h // 16), max(1, w // 16), 3), dtype=np.uint8)
    img = np.kron(blocks, np.ones((16, 16, 1), dtype=np.uint8))[:h, :w]
    img = np.pad(img, ((0, max(0, h - img.shape[0])), (0, max(0, w - img.shape[1])), (0, 0)), mode="edge")
    noise = rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8))


def image_input(data: bytes, w: int = 1, h: int = 1) -> ImageInput:
    meta = ImageMetadata(image_id="IMG-T", inspection_id="INS-T", file_name="t.png", mime_type="image/png",
                         size_bytes=len(data), width=w, height=h, sha256="0" * 64, storage_backend="test",
                         storage_reference="mem", created_at=utcnow())
    return ImageInput(metadata=meta, load_bytes=lambda: data)
