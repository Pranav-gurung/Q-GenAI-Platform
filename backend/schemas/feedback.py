"""Operator feedback: InspectionResult + HumanFeedback -> FeedbackRecord (input for the learning loop).

Part 6 adds a structured review of each AI layer. Corrections reuse the existing fields:
  decision_correct=false  -> `outcome` is the corrected decision
  defect_correct=false    -> `defect_types` is the corrected defect set
  stage_correct=false     -> `suspected_stage` is the corrected first-observed stage (operator ground truth, not a
                             causal claim)
`*_correct=true` confirms the AI value (the matching field may be omitted). The AI value is snapshotted server-side
next to the operator value; feedback never edits the AI result. Nothing here trains or changes a model or rule.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
        "operator_id": "OP-17", "decision_correct": True, "defect_correct": True, "stage_correct": False,
        "suspected_stage": "PICKING", "rca_useful": True, "action_useful": True,
        "comment": "Leak was visible during picking inspection."}})

    operator_id: ExternalId
    outcome: DecisionOutcome | None = Field(None, description="The operator's ground-truth verdict. Required unless "
                                                                "decision_correct=true (then the AI decision is used).")
    defect_types: list[DefectType] = Field(default_factory=list, max_length=20, description=(
        "The operator's defect types (the correction when defect_correct=false)."))
    corrected_detections: list[CorrectedDetection] = Field(default_factory=list, max_length=100)
    suspected_stage: FulfillmentStage | None = Field(None, description=(
        "The stage where the operator says the defect was first observable (the correction when stage_correct=false). "
        "Operator ground truth for evaluation; it does not change the system's attribution."))
    comment: str | None = Field(None, max_length=1000, description="Free text. Untrusted data, never instructions.")
    decision_correct: bool | None = Field(None, description="Was the automated PASS/FAIL/REVIEW decision correct?")
    defect_correct: bool | None = Field(None, description="Were the CV-detected defect types correct?")
    stage_correct: bool | None = Field(None, description="Was the first observed stage (pipeline attribution) correct?")
    rca_useful: bool | None = Field(None, description="Was the GenAI root-cause analysis useful?")
    action_useful: bool | None = Field(None, description="Were the recommended corrective actions useful?")

    @model_validator(mode="after")
    def _corrections_are_complete(self) -> "FeedbackCreate":
        if self.outcome is None and self.decision_correct is not True:
            raise ValueError("outcome is required unless decision_correct is true")
        if self.stage_correct is False and self.suspected_stage is None:
            raise ValueError("suspected_stage (the corrected stage) is required when stage_correct is false")
        return self


class FeedbackCorrection(BaseModel):
    """One layer where the operator disagreed with the AI. Both values are kept."""
    layer: str = Field(description="DECISION | DEFECT | STAGE")
    ai_value: str | list[str] | None
    operator_value: str | list[str] | None


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
    item_id: str | None = None
    order_id: str | None = None
    decision_correct: bool | None = None
    defect_correct: bool | None = None
    stage_correct: bool | None = None
    rca_useful: bool | None = None
    action_useful: bool | None = None
    ai_defect_types: list[DefectType] | None = Field(None, description="CV defect types at review time (null: no "
                                                                        "completed CV result).")
    ai_stage: FulfillmentStage | None = Field(None, description="First observed stage shown at review time.")
    ai_snapshot: dict = Field(default_factory=dict, description="What the AI showed at review time: decision, "
                                                                 "defects, stage attribution, RCA/RAG status.")
    corrections: list[FeedbackCorrection] = Field(default_factory=list,
                                                  description="Layers marked incorrect, AI value vs operator value.")
    learning_status: LearningStatus
    created_at: datetime
