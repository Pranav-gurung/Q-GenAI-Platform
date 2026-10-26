"""Upload validation, decoding and annotation drawing."""
import io
from dataclasses import dataclass
from pathlib import PurePath

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.utils.hashing import sha256_hex

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}  # detected from file content, not the name


class ImageValidationError(ValueError):
    """Raised with a user-safe message when an upload is not acceptable."""


@dataclass
class ValidatedImage:
    rgb: np.ndarray  # H x W x 3, uint8, RGB order
    width: int
    height: int
    sha256: str
    format: str


def validate_and_decode(
    data: bytes, filename: str | None, max_bytes: int, max_pixels: int
) -> ValidatedImage:
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"The file is too large ({len(data) / 1_048_576:.1f} MB). "
            f"Maximum allowed is {max_bytes / 1_048_576:.1f} MB."
        )

    ext = PurePath(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Only .jpg, .jpeg and .png images are accepted.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise ImageValidationError("The file content is not a JPEG or PNG image.")
            if width * height > max_pixels:
                raise ImageValidationError(
                    f"Image resolution {width}x{height} exceeds the allowed pixel count."
                )
            probe.verify()  # structural integrity check (does not decode pixels)

        # verify() leaves the object unusable; reopen to actually decode.
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)  # respect camera orientation
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ImageValidationError("The file could not be decoded as a valid image.") from None

    return ValidatedImage(
        rgb=rgb, width=rgb.shape[1], height=rgb.shape[0], sha256=sha256_hex(data), format=fmt
    )


# A small fixed palette (BGR) so the same class always gets the same colour.
_PALETTE = [
    (56, 56, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
    (23, 204, 146), (134, 219, 61), (211, 188, 0), (255, 149, 0), (255, 56, 132),
]


def draw_detections(rgb: np.ndarray, defects: list[dict]) -> bytes:
    """Return PNG bytes of the image with bounding boxes and labels drawn."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    thickness = max(2, round(0.002 * (bgr.shape[0] + bgr.shape[1])))
    font_scale = max(0.5, thickness / 4)
    for d in defects:
        color = _PALETTE[d.get("class_id", 0) % len(_PALETTE)]
        b = d["bbox"]
        p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(bgr, p1, p2, color, thickness)
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(p1[1] - th - base - 4, 0)
        cv2.rectangle(bgr, (p1[0], top), (p1[0] + tw + 4, top + th + base + 4), color, -1)
        cv2.putText(bgr, label, (p1[0] + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode annotated image")
    return buf.tobytes()
