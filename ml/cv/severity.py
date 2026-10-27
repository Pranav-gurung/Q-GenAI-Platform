"""Deterministic severity mapping for a single detection.

    severity = base_severity(class)                      # from taxonomy.yaml
    if box_area / image_area >= CV_SEVERITY_LARGE_AREA_RATIO:  severity += 1 level
    if class is safety_critical:                         severity = max(severity, HIGH)
    (capped at CRITICAL)

Confidence deliberately does NOT change severity: severity describes the impact of the defect,
confidence describes how sure the model is. The decision rules combine both.
No randomness, no ML, no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.domain.enums import Severity
from backend.domain.severity import escalate, max_severity
from ml.cv.taxonomy import DefectClass


@dataclass(frozen=True)
class SeverityPolicy:
    large_area_ratio: float = 0.20

    def __post_init__(self) -> None:
        if not 0 < self.large_area_ratio <= 1:
            raise ValueError("large_area_ratio must be in (0, 1]")


def compute_severity(defect: DefectClass, area_ratio: float, policy: SeverityPolicy) -> Severity:
    if not 0 <= area_ratio <= 1:
        raise ValueError(f"area_ratio must be in [0, 1], got {area_ratio}")
    level = defect.base_severity
    if area_ratio >= policy.large_area_ratio:
        level = escalate(level)
    if defect.safety_critical:
        level = max_severity([level, Severity.HIGH])
    return level
