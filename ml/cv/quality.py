"""Image quality gate (QUALITY_GATE_ENGINE=basic).

File-level validation (type, size, corruption) already happened at upload
(backend/services/image_validation.py). This gate only judges whether the *content* is usable
for visual inspection. It is deliberately conservative: it rejects images that are unusable,
not images that are merely imperfect.

Metrics (all deterministic):
  width, height            pixels after EXIF orientation
  aspect_ratio             long side / short side
  blur_variance            variance of the 4-neighbour Laplacian on the grayscale image resized
                           so its long side is 512 px (scale-normalised). Low = blurry.
  brightness               mean luminance 0-255

Issues -> acceptable=False:
  LOW_RESOLUTION     width < CV_MIN_IMAGE_WIDTH or height < CV_MIN_IMAGE_HEIGHT  (default 128)
  EXTREME_ASPECT     aspect_ratio > CV_MAX_ASPECT_RATIO                          (default 6)
  BLURRY             blur_variance < CV_BLUR_THRESHOLD                           (default 15)
  TOO_DARK           brightness < CV_MIN_BRIGHTNESS                              (default 20)
  TOO_BRIGHT         brightness > CV_MAX_BRIGHTNESS                              (default 240)
"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from backend.core.config import Settings
from backend.domain.enums import ModuleStatus
from backend.pipeline.context import ImageInput
from backend.pipeline.interfaces import ImageQualityGate
from backend.schemas.results import ImageQualityResult
from ml.cv.preprocessing import decode_image

QUALITY_GATE_VERSION = "1"
_BLUR_LONG_SIDE = 512


def laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var()) if lap.size else 0.0


def measure(rgb: np.ndarray) -> dict[str, float]:
    h, w = rgb.shape[:2]
    gray_img = Image.fromarray(rgb).convert("L")
    scale = _BLUR_LONG_SIDE / max(w, h)
    if scale < 1:
        gray_img = gray_img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    gray = np.asarray(gray_img)
    return {"width": float(w), "height": float(h), "aspect_ratio": round(max(w, h) / min(w, h), 3),
            "blur_variance": round(laplacian_variance(gray), 2),
            "brightness": round(float(np.asarray(Image.fromarray(rgb).convert("L")).mean()), 2)}


def find_issues(metrics: dict[str, float], settings: Settings) -> list[str]:
    issues = []
    if metrics["width"] < settings.cv_min_image_width or metrics["height"] < settings.cv_min_image_height:
        issues.append("LOW_RESOLUTION")
    if metrics["aspect_ratio"] > settings.cv_max_aspect_ratio:
        issues.append("EXTREME_ASPECT")
    if metrics["blur_variance"] < settings.cv_blur_threshold:
        issues.append("BLURRY")
    if metrics["brightness"] < settings.cv_min_brightness:
        issues.append("TOO_DARK")
    if metrics["brightness"] > settings.cv_max_brightness:
        issues.append("TOO_BRIGHT")
    return issues


class BasicImageQualityGate(ImageQualityGate):
    name = "basic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def health(self) -> dict:
        s = self.settings
        return {"available": True, "version": QUALITY_GATE_VERSION,
                "thresholds": {"min_width": s.cv_min_image_width, "min_height": s.cv_min_image_height,
                               "max_aspect_ratio": s.cv_max_aspect_ratio, "blur": s.cv_blur_threshold,
                               "min_brightness": s.cv_min_brightness, "max_brightness": s.cv_max_brightness}}

    def assess(self, image: ImageInput) -> ImageQualityResult:
        t0 = time.perf_counter()
        metrics = measure(decode_image(image.load_bytes()))
        issues = find_issues(metrics, self.settings)
        return ImageQualityResult(
            module=self.name, module_version=QUALITY_GATE_VERSION, status=ModuleStatus.COMPLETED,
            acceptable=not issues, issues=issues, metrics=metrics,
            message="Image rejected: " + ", ".join(issues) if issues else "Image usable for inspection",
            processing_time_ms=(time.perf_counter() - t0) * 1000)
