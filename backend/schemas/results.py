"""Result contracts produced by pipeline modules.

Every result extends `ModuleResult`, which records WHICH module produced it and whether it
produced a real result (status=COMPLETED) or not (NOT_IMPLEMENTED / SKIPPED / FAILED).

Guard against fake output: if status != COMPLETED, all payload fields must be empty
(None / [] / {}). A placeholder module therefore *cannot* emit a confidence score or a
detection, and consumers can always tell a real result from a missing one.
"""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.domain.enums import (ActionPriority, DecisionOutcome, DecisionSource, DefectType,
                                  ModuleStatus, Severity)
from backend.domain.fulfillment import FulfillmentStage

Confidence = float  # constrained per-field with ge=0, le=1


class ModuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Names of the fields that carry the module's actual output.
    payload_fields: ClassVar[tuple[str, ...]] = ()

    module: str = Field(description="Implementation that produced this result, e.g. 'stub', 'yolov8'.")
    module_version: str | None = Field(None, description="Version of the implementation / model weights.")
    status: ModuleStatus
    message: str | None = Field(None, max_length=500)
    processing_time_ms: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _no_payload_unless_completed(self) -> "ModuleResult":
        if self.status is not ModuleStatus.COMPLETED:
            filled = [f for f in self.payload_fields if getattr(self, f) not in (None, [], {})]
            if filled:
                raise ValueError(f"{type(self).__name__} with status {self.status.value} "
                                 f"must not carry result values: {filled}")
            if self.module_version is not None and self.status is ModuleStatus.NOT_IMPLEMENTED:
                raise ValueError("NOT_IMPLEMENTED results must not claim a module_version")
        return self

    @property
    def is_real(self) -> bool:
        return self.status is ModuleStatus.COMPLETED

    @classmethod
    def not_implemented(cls, module: str, message: str):
        return cls(module=module, status=ModuleStatus.NOT_IMPLEMENTED, message=message)

    @classmethod
    def skipped(cls, module: str, message: str):
        return cls(module=module, status=ModuleStatus.SKIPPED, message=message)

    @classmethod
    def failed(cls, module: str, message: str):
        return cls(module=module, status=ModuleStatus.FAILED, message=message)

    @classmethod
    def unavailable(cls, module: str, message: str, module_version: str | None = None):
        return cls(module=module, module_version=module_version, status=ModuleStatus.UNAVAILABLE, message=message)


# ---------------------------------------------------------------- image quality (Part 3)
class ImageQualityResult(ModuleResult):
    payload_fields: ClassVar = ("acceptable", "issues", "metrics")

    acceptable: bool | None = Field(None, description="False -> image must be retaken; pipeline stops.")
    issues: list[str] = Field(default_factory=list, examples=[["BLUR", "OVEREXPOSED"]])
    metrics: dict[str, float] = Field(default_factory=dict, examples=[{"blur_variance": 112.4}])


# ---------------------------------------------------------------- computer vision (Part 2)
class BoundingBox(BaseModel):
    """Pixel coordinates in the ORIGINAL image."""
    model_config = ConfigDict(extra="forbid")
    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(gt=0)
    y_max: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "BoundingBox":
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must satisfy x_min < x_max and y_min < y_max")
        return self


class Detection(BaseModel):
    """One model detection (one box).

    Coordinates: `bbox` is in PIXELS of the inspected image (after EXIF orientation is applied),
    origin top-left, x to the right, y down, (x_min, y_min, x_max, y_max). `area_ratio` is the box
    area divided by the image area (0-1).
    """
    model_config = ConfigDict(extra="forbid")
    defect_type: DefectType
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox | None = None
    severity: Severity | None = None
    mask_ref: str | None = None
    model_class: str | None = Field(None, description="Raw class name emitted by the model.")
    area_ratio: float | None = Field(None, ge=0, le=1)


class Defect(BaseModel):
    """A defect aggregated over one or more detections."""
    model_config = ConfigDict(extra="forbid")
    defect_type: DefectType
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    detection_count: int = Field(ge=1)


class CVInferenceMetadata(BaseModel):
    """How a CV result was produced. Present only when inference actually ran."""
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model_name: str = Field(examples=["yolov8n"])
    model_version: str
    model_framework: str = Field(examples=["ultralytics 8.4.190 / torch 2.x"])
    model_artifact: str = Field(description="Artifact file name and short content hash (no absolute paths).")
    inference_device: str = Field(description="Device actually used, e.g. 'cpu' or 'cuda:0'.")
    image_size: int = Field(description="Model input size (letterboxed square, pixels).")
    image_width: int = Field(gt=0, description="Inspected image width (pixels, after EXIF orientation).")
    image_height: int = Field(gt=0)
    confidence_threshold: float = Field(ge=0, le=1)
    iou_threshold: float = Field(ge=0, le=1)
    max_detections: int = Field(gt=0)
    preprocess_ms: float | None = Field(None, ge=0)
    inference_time_ms: float = Field(ge=0, description="Model forward pass only.")
    postprocess_ms: float | None = Field(None, ge=0)
    raw_detection_count: int = Field(ge=0, description="Model boxes before Q-GenAI confidence filtering.")


class CVResult(ModuleResult):
    payload_fields: ClassVar = ("detections", "defects", "classification", "overall_confidence", "inference")

    detections: list[Detection] = Field(default_factory=list)
    defects: list[Defect] = Field(default_factory=list)
    classification: str | None = Field(None, description="Image-level summary: 'DEFECTS_DETECTED' or "
                                                          "'NO_DEFECTS_DETECTED' (only when inference ran).")
    overall_confidence: float | None = Field(None, ge=0, le=1, description="Highest detection confidence; "
                                                                          "null when nothing was detected.")
    inference: CVInferenceMetadata | None = None


# ---------------------------------------------------------------- decision
class QualityDecision(ModuleResult):
    payload_fields: ClassVar = ("outcome", "reason", "confidence", "severity", "decision_source")

    outcome: DecisionOutcome | None = None
    reason: str | None = Field(None, max_length=500)
    confidence: float | None = Field(None, ge=0, le=1)
    severity: Severity | None = None
    decision_source: DecisionSource | None = None

    @model_validator(mode="after")
    def _completed_needs_outcome(self) -> "QualityDecision":
        if self.status is ModuleStatus.COMPLETED and (self.outcome is None or self.decision_source is None):
            raise ValueError("a COMPLETED decision needs outcome and decision_source")
        return self


# ---------------------------------------------------------------- fulfillment analysis (Part 4)
class PipelineAnalysis(ModuleResult):
    payload_fields: ClassVar = ("suspected_stage", "stage_likelihoods", "evidence", "confidence")

    suspected_stage: FulfillmentStage | None = None
    stage_likelihoods: dict[FulfillmentStage, float] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = Field(None, ge=0, le=1)


# ---------------------------------------------------------------- RCA (Part 5)
class RootCause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cause: str = Field(max_length=500)
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    stage: FulfillmentStage | None = None


class RCAResult(ModuleResult):
    payload_fields: ClassVar = ("possible_causes", "recommended_investigation", "confidence")

    possible_causes: list[RootCause] = Field(default_factory=list)
    recommended_investigation: list[str] = Field(default_factory=list)
    confidence: float | None = Field(None, ge=0, le=1)


# ---------------------------------------------------------------- corrective action (Part 5)
class CorrectiveAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str = Field(max_length=200, examples=["Quarantine affected batch"])
    priority: ActionPriority
    responsible_stage: FulfillmentStage
    instructions: list[str] = Field(default_factory=list)
    sop_reference: str | None = None


class ActionRecommendation(ModuleResult):
    payload_fields: ClassVar = ("actions",)

    actions: list[CorrectiveAction] = Field(default_factory=list)
