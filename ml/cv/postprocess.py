"""Model output -> validated Q-GenAI detections.

Framework-independent: works on plain arrays so it can be unit-tested and reused by a future
ONNX Runtime engine. Steps: validate shapes/values -> map class ids to taxonomy -> per-class
confidence filtering -> clip boxes to the image -> reject degenerate boxes -> severity -> sort/cap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.domain.severity import max_severity
from backend.schemas.results import BoundingBox, Defect, Detection
from ml.cv.severity import SeverityPolicy, compute_severity
from ml.cv.taxonomy import DefectClass

# Boxes may overshoot the image by a few pixels due to letterbox rounding; more than this is malformed.
_CLIP_TOLERANCE_PX = 2.0
_SCORE_EPS = 1e-6


class MalformedPredictionError(ValueError):
    """The model returned output that cannot be trusted (wrong shape, NaN, unknown class, ...)."""


@dataclass(frozen=True)
class RawPredictions:
    boxes_xyxy: np.ndarray  # (N, 4) pixel coordinates in the ORIGINAL image
    scores: np.ndarray      # (N,)
    class_ids: np.ndarray   # (N,)

    @classmethod
    def empty(cls) -> "RawPredictions":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))

    def __len__(self) -> int:
        return int(self.scores.shape[0])


def _validate(raw: RawPredictions) -> None:
    n = raw.scores.shape[0] if raw.scores.ndim == 1 else -1
    if raw.boxes_xyxy.ndim != 2 or raw.boxes_xyxy.shape[1] != 4 or raw.boxes_xyxy.shape[0] != n \
            or raw.class_ids.shape != (n,):
        raise MalformedPredictionError(f"inconsistent prediction shapes: boxes {raw.boxes_xyxy.shape}, "
                                       f"scores {raw.scores.shape}, classes {raw.class_ids.shape}")
    if not (np.isfinite(raw.boxes_xyxy).all() and np.isfinite(raw.scores).all()):
        raise MalformedPredictionError("non-finite values in model output")
    if ((raw.scores < 0) | (raw.scores > 1)).any():
        raise MalformedPredictionError("confidence outside [0, 1]")
    if not np.all(np.equal(np.mod(raw.class_ids, 1), 0)):
        raise MalformedPredictionError("non-integer class ids")


def to_detections(raw: RawPredictions, *, image_width: int, image_height: int,
                  class_map: dict[int, DefectClass], confidence_threshold: float, max_detections: int,
                  severity_policy: SeverityPolicy) -> list[Detection]:
    _validate(raw)
    image_area = float(image_width * image_height)
    out: list[Detection] = []
    for box, score, cls in zip(raw.boxes_xyxy.astype(float), raw.scores.astype(float), raw.class_ids.astype(int)):
        spec = class_map.get(int(cls))
        if spec is None:
            raise MalformedPredictionError(f"model emitted unknown class id {int(cls)}")
        threshold = spec.min_confidence if spec.min_confidence is not None else confidence_threshold
        if score < threshold - _SCORE_EPS:  # model scores are float32: 0.35f == 0.3499999
            continue
        x1, y1, x2, y2 = box
        if (x1 < -_CLIP_TOLERANCE_PX or y1 < -_CLIP_TOLERANCE_PX or x2 > image_width + _CLIP_TOLERANCE_PX
                or y2 > image_height + _CLIP_TOLERANCE_PX):
            raise MalformedPredictionError(f"box {box.tolist()} outside image {image_width}x{image_height}")
        x1, x2 = np.clip([x1, x2], 0, image_width)
        y1, y2 = np.clip([y1, y2], 0, image_height)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            raise MalformedPredictionError(f"degenerate box {box.tolist()}")
        area_ratio = min(1.0, (x2 - x1) * (y2 - y1) / image_area)
        out.append(Detection(
            defect_type=spec.defect_type, confidence=round(score, 4), model_class=spec.name,
            bbox=BoundingBox(x_min=round(float(x1), 1), y_min=round(float(y1), 1),
                             x_max=round(float(x2), 1), y_max=round(float(y2), 1)),
            area_ratio=round(area_ratio, 4), severity=compute_severity(spec, area_ratio, severity_policy)))
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out[:max_detections]


def aggregate_defects(detections: list[Detection]) -> list[Defect]:
    """One Defect per defect type: max severity, max confidence, number of boxes."""
    groups: dict = {}
    for d in detections:
        groups.setdefault(d.defect_type, []).append(d)
    defects = [Defect(defect_type=t, severity=max_severity(d.severity for d in ds),
                      confidence=max(d.confidence for d in ds), detection_count=len(ds))
               for t, ds in groups.items()]
    return sorted(defects, key=lambda d: d.confidence, reverse=True)
