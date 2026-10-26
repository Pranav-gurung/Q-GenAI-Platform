"""Shared helpers for the CV command-line tools."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.core.config import Settings, get_settings
from backend.core.logging import configure_logging
from backend.pipeline.context import ImageInput
from backend.schemas.common import utcnow
from backend.schemas.image import ImageMetadata
from backend.services.image_validation import validate_image


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", type=Path, help="weights file (default: CV_MODEL_PATH)")
    p.add_argument("--device", help="auto | cpu | cuda | cuda:N (default: CV_DEVICE)")
    p.add_argument("--conf", type=float, help="confidence threshold (default: CV_CONFIDENCE_THRESHOLD)")
    p.add_argument("--imgsz", type=int, help="inference image size (default: CV_IMAGE_SIZE or training size)")


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    configure_logging("WARNING", settings.log_format, stream=sys.stderr)  # stdout is reserved for JSON output
    logging.getLogger("ml.cv").setLevel(logging.INFO)
    update = {k: v for k, v in {"cv_model_path": args.model, "cv_device": args.device,
                                "cv_confidence_threshold": args.conf, "cv_image_size": args.imgsz}.items()
              if v is not None}
    return Settings(**{**settings.model_dump(), **update}) if update else settings


def image_input_from_file(path: Path, settings: Settings) -> ImageInput:
    """Validate a local image exactly like an API upload and wrap it as pipeline input."""
    data = Path(path).read_bytes()
    img = validate_image(data, Path(path).name, None, allowed_types=settings.allowed_image_types,
                         max_bytes=settings.max_upload_bytes, max_pixels=settings.max_image_pixels)
    meta = ImageMetadata(image_id="LOCAL", inspection_id="LOCAL", file_name=img.file_name, mime_type=img.mime_type,
                         size_bytes=img.size_bytes, width=img.width, height=img.height, sha256=img.sha256,
                         storage_backend="file", storage_reference=str(path), created_at=utcnow())
    return ImageInput(metadata=meta, load_bytes=lambda: data)
