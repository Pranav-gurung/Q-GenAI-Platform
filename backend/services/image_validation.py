"""Upload validation: size, declared type, actual decoded format, integrity, pixel bomb guard.

The file *content* is authoritative: the extension and declared Content-Type must agree
with what Pillow actually decodes. Image *quality* (blur, exposure ...) is NOT checked here;
that is the Image Quality Gate's job (Part 3).
"""
from __future__ import annotations

import hashlib
import io
import re
import warnings
from dataclasses import dataclass
from pathlib import PurePath

from PIL import Image, UnidentifiedImageError

from backend.core.exceptions import InvalidImageError

# Pillow format name -> (canonical type, mime, accepted extensions)
FORMATS: dict[str, tuple[str, str, frozenset[str]]] = {
    "JPEG": ("jpeg", "image/jpeg", frozenset({".jpg", ".jpeg"})),
    "PNG": ("png", "image/png", frozenset({".png"})),
    "WEBP": ("webp", "image/webp", frozenset({".webp"})),
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    file_name: str
    image_type: str  # jpeg | png | webp
    mime_type: str
    extension: str
    width: int
    height: int
    sha256: str

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def sanitize_filename(name: str | None) -> str:
    base = PurePath(name or "upload").name  # drop any directory components
    base = _SAFE_NAME.sub("_", base).strip("._") or "upload"
    if len(base) <= 100:
        return base
    stem, dot, ext = base.rpartition(".")
    if dot and 0 < len(ext) <= 10:  # keep the extension: it is checked against the decoded content
        return stem[:100 - len(ext) - 1] + "." + ext
    return base[:100]


def validate_image(data: bytes, filename: str | None, content_type: str | None, *,
                   allowed_types: list[str], max_bytes: int, max_pixels: int) -> ValidatedImage:
    if not data:
        raise InvalidImageError("Uploaded file is empty", code="EMPTY_FILE")
    if len(data) > max_bytes:
        raise InvalidImageError(f"Image exceeds maximum size of {max_bytes // (1024 * 1024)} MB",
                                code="IMAGE_TOO_LARGE", details={"max_bytes": max_bytes})

    file_name = sanitize_filename(filename)
    ext = PurePath(file_name).suffix.lower()
    allowed_exts = {e for fmt in FORMATS.values() if fmt[0] in allowed_types for e in fmt[2]}
    if ext not in allowed_exts:
        raise InvalidImageError(f"Unsupported file extension '{ext or '(none)'}'", code="UNSUPPORTED_IMAGE_TYPE",
                                details={"allowed_extensions": sorted(allowed_exts)})

    # Decode. verify() catches truncation/corruption; a second open is required after verify().
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                width, height = probe.size
                if width * height > max_pixels:
                    raise InvalidImageError("Image dimensions too large", code="IMAGE_TOO_LARGE",
                                            details={"max_pixels": max_pixels})
                probe.verify()
            with Image.open(io.BytesIO(data)) as img:
                img.load()  # full decode: catches truncated pixel data that verify() can miss
    except InvalidImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImageError("Image dimensions too large", code="IMAGE_TOO_LARGE") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("File is not a valid image or is corrupt", code="CORRUPT_IMAGE") from exc
    except Exception as exc:  # untrusted bytes: any other decoder / plugin failure is still a bad file, never a 500
        raise InvalidImageError("File is not a valid image or is corrupt", code="CORRUPT_IMAGE") from exc

    if fmt not in FORMATS or FORMATS[fmt][0] not in allowed_types:
        raise InvalidImageError(f"Unsupported image format '{fmt}'", code="UNSUPPORTED_IMAGE_TYPE",
                                details={"allowed_types": allowed_types})
    image_type, mime, exts = FORMATS[fmt]
    if ext not in exts:
        raise InvalidImageError(f"File extension '{ext}' does not match image content ({image_type})",
                                code="UNSUPPORTED_IMAGE_TYPE")
    declared = (content_type or "").split(";")[0].strip().lower()
    if declared and declared not in (mime, "application/octet-stream"):
        raise InvalidImageError(f"Declared content type '{declared}' does not match image content ({mime})",
                                code="UNSUPPORTED_IMAGE_TYPE")
    if width <= 0 or height <= 0:
        raise InvalidImageError("Image has no pixels", code="CORRUPT_IMAGE")

    return ValidatedImage(data=data, file_name=file_name, image_type=image_type, mime_type=mime,
                          extension=sorted(exts)[0] if image_type != "jpeg" else ".jpg",
                          width=width, height=height, sha256=hashlib.sha256(data).hexdigest())
