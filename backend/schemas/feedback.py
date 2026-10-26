"""Operator feedback: InspectionResult + HumanFeedback -> FeedbackRecord (input for active learning)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.domain.enums import DecisionOutcome, DefectType, LearningStatus
from backend.domain.fulfillment import FulfillmentStage
from backend.schemas.common import ExternalId, StrictModel
from backend.schemas.results import BoundingBox


class CorrectedDetection(StrictModel):
    """An operator-drawn/confirmed defect region (future annotation UI)."""
    defect_type: DefectType
    bbox: BoundingBox | None = None


class FeedbackCreate(StrictModel):
    """Request body for POST /api/inspections/{inspection_id}/feedback."""
    model_config = StrictModel.model_config | ConfigDict(json_schema_extra={"example": {
        "operator_id": "OP-17", "outcome": "FAIL", "defect_types": ["LEAKING_PACKAGE"],
        "suspected_stage": "PICKING", "comment": "Milk pouch leaking at the seam"}})

    operator_id: ExternalId
    outcome: DecisionOutcome = Field(description="The operator's ground-truth verdict.")
    defect_types: list[DefectType] = Field(default_factory=list, max_length=20)
    corrected_detections: list[CorrectedDetection] = Field(default_factory=list, max_length=100)
    suspected_stage: FulfillmentStage | None = Field(None, description="Where the operator thinks it happened.")
    comment: str | None = Field(None, max_length=1000)


class FeedbackRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    feedback_id: str
    inspection_id: str
    result_id: str | None
    operator_id: str
    outcome: DecisionOutcome
    defect_types: list[DefectType]
    corrected_detections: list[CorrectedDetection]
    suspected_stage: FulfillmentStage | None
    comment: str | None
    system_outcome: DecisionOutcome | None = Field(description="What the system decided; null if it made no "
                                                                "automated decision.")
    is_correction: bool = Field(description="True if the operator disagreed with a real system decision.")
    model_versions: dict[str, str] = Field(description="Snapshot of the models that produced the result.")
    learning_status: LearningStatus
    created_at: datetime
