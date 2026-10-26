from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal["PASS", "FAIL", "REVIEW"]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DefectOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


class OverrideOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    actor_id: str
    actor_role: str
    old_decision: Decision
    new_decision: Decision
    reason: str
    created_at: datetime


class AIResult(BaseModel):
    """The ORIGINAL model output and automatic decision - never modified."""

    inspection_status: Decision
    decision_reasons: list[str]
    defect_count: int
    defects: list[DefectOut]


class InspectionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    inspection_id: str
    timestamp: datetime
    inspected_by: str
    site: str
    model_status: str
    model_name: str | None
    model_version: str | None
    model_sha256: str | None
    policy_version: int
    confidence_threshold: float
    threshold_overridden: bool
    processing_time_ms: float
    inference_time_ms: float | None
    image_width: int
    image_height: int
    image_sha256: str
    # Flat convenience fields (mirror ai_result) for simple clients
    inspection_status: Decision
    defect_count: int
    defects: list[DefectOut]
    ai_result: AIResult
    final_decision: Decision
    overrides: list[OverrideOut]
    annotated_image_available: bool
    audit_event_ids: list[str] = []


class InspectionSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    inspection_id: str
    timestamp: datetime
    inspected_by: str
    site: str
    ai_decision: Decision
    final_decision: Decision
    defect_count: int
    model_version: str | None
    has_override: bool


class InspectionList(BaseModel):
    total: int
    items: list[InspectionSummary]


class OverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_decision: Decision
    reason: str = Field(min_length=10, max_length=2000,
                        description="Mandatory justification (min. 10 characters).")


class RCAPayload(BaseModel):
    """Structured data handed to Section C (root-cause analysis).

    Contains NO image data and NO personal data (no user names or ids).
    """

    model_config = ConfigDict(protected_namespaces=())

    inspection_id: str
    timestamp: datetime
    site: str
    model_version: str | None
    model_sha256: str | None
    policy_version: int
    confidence_threshold: float
    ai_decision: Decision
    final_decision: Decision
    defects: list[DefectOut]
    image_width: int
    image_height: int


class RCADefectItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_: str = Field(alias="class", description="Supported defect class name")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score in [0.0, 1.0]")
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4,
                                     description="[x_min, y_min, x_max, y_max] in original-image pixels")
    mask_ref: str | None = Field(default=None, description="Optional mask reference")


class RCARequestOut(BaseModel):
    """Schema strictly conforming to Module C RCARequest (schemas/rca_request.schema.json)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    inspection_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")
    image_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")
    model_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")
    status: str | None = Field(default=None, max_length=64)
    defects: list[RCADefectItem]
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)
    part_type: str | None = Field(default=None, max_length=300)
    material: str | None = Field(default=None, max_length=300)
    production_process: str | None = Field(default=None, max_length=300)
    inspection_context: str | None = Field(default=None, max_length=300)


class PolicyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: int
    created_at: datetime
    created_by: str | None
    change_reason: str
    default_confidence_threshold: float
    operator_min_threshold: float
    operator_max_threshold: float
    fail_confidence: float
    fail_classes: list[str]
    ignored_classes: list[str]
    max_allowed_defects: int
    require_approved_model: bool


class PolicyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_confidence_threshold: float = Field(gt=0, lt=1)
    operator_min_threshold: float = Field(gt=0, lt=1)
    operator_max_threshold: float = Field(gt=0, lt=1)
    fail_confidence: float = Field(gt=0, lt=1)
    fail_classes: list[str] = Field(default_factory=list, max_length=100)
    ignored_classes: list[str] = Field(default_factory=list, max_length=100)
    max_allowed_defects: int = Field(0, ge=0, le=1000)
    require_approved_model: bool = False
    change_reason: str = Field(min_length=10, max_length=2000)

    @model_validator(mode="after")
    def _ranges(self):
        if not (self.operator_min_threshold <= self.default_confidence_threshold
                <= self.operator_max_threshold):
            raise ValueError("Require operator_min <= default_confidence_threshold <= operator_max.")
        if set(self.fail_classes) & set(self.ignored_classes):
            raise ValueError("A class cannot be both a fail class and an ignored class.")
        return self
