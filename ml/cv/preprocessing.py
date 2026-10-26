"""Image preprocessing.

INFERENCE (deterministic, never alters content in ways that could hide defects):
  bytes -> decode (Pillow) -> apply EXIF orientation -> RGB uint8 HxWx3 -> BGR for the YOLO predictor.
  The YOLO predictor then letterboxes to CV_IMAGE_SIZE (aspect ratio preserved, grey padding),
  scales to [0, 1] float32 CHW, and maps boxes back to original-image pixels. No augmentation,
  no sharpening / denoising / colour correction.

TRAINING (stochastic, only in ml/cv/train.py): TRAINING_AUGMENTATION below. Kept moderate:
  geometric flips/scale/translate and mild colour jitter. No transforms that would erase defect
  evidence (e.g. heavy blur, cutout over the product).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class PreprocessingError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to an RGB uint8 array (H, W, 3) with EXIF orientation applied."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreprocessingError("cannot decode image") from exc
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise PreprocessingError(f"unexpected decoded image shape {arr.shape}")
    return arr


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Ultralytics treats numpy inputs as BGR (OpenCV convention)."""
    return np.ascontiguousarray(rgb[:, :, ::-1])


def to_model_input(data: bytes) -> tuple[np.ndarray, int, int]:
    """bytes -> (BGR array for the predictor, width, height)."""
    rgb = decode_image(data)
    return rgb_to_bgr(rgb), rgb.shape[1], rgb.shape[0]


# Ultralytics hyper-parameter names. Mosaic helps small datasets; values are moderate on purpose.
TRAINING_AUGMENTATION: dict[str, float] = {
    "hsv_h": 0.015, "hsv_s": 0.4, "hsv_v": 0.3,
    "degrees": 0.0, "translate": 0.1, "scale": 0.3, "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0,
}
