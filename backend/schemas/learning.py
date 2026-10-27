"""Feedback error analysis and learning-candidate export (Part 6).

Deterministic aggregation over stored operator feedback. Nothing here trains a model, edits a rule or changes
production behaviour: the output is for people (ML engineers, QA) to review and curate.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from backend.domain.enums import DecisionOutcome, DefectType, LearningStatus
from backend.domain.fulfillment import FulfillmentStage

LEARNING_LOOP = ["AI output", "Operator feedback", "Error analysis", "Candidate training data",
                 "Human / ML engineer review", "Future model version"]


class LayerStats(BaseModel):
    reviewed: int = 0
    confirmed: int = 0
    corrected: int = 0
    correction_rate: float | None = Field(None, description="corrected / reviewed (null when nothing reviewed).")


class BreakdownStats(LayerStats):
    by_ai_value: dict[str, LayerStats] = Field(default_factory=dict,
                                               description="Per AI defect type / AI stage / AI decision.")
    transitions: dict[str, int] = Field(default_factory=dict, description='"AI -> operator" correction counts.')


class UsefulnessStats(BaseModel):
    rated: int = 0
    useful: int = 0
    not_useful: int = 0
    useful_rate: float | None = None


class LearningLoopStatus(BaseModel):
    flow: list[str] = Field(default_factory=lambda: list(LEARNING_LOOP))
    automatic_retraining: bool = False
    automatic_rule_changes: bool = False
    automatic_deployment: bool = False
    note: str = ("Feedback is recorded for future system improvement. Candidates require human curation before any "
                 "dataset, evaluation or model version uses them.")


class FeedbackAnalysis(BaseModel):
    generated_at: datetime
    scope: str = "latest feedback per inspection"
    feedback_submissions: int
    inspections_reviewed: int
    decisions: BreakdownStats = Field(description="Operator verdict vs the automated decision (overrides).")
    defects: BreakdownStats = Field(description="defect_correct reviews, by AI defect type.")
    stages: BreakdownStats = Field(description="stage_correct reviews, by AI first observed stage.")
    rca: UsefulnessStats
    actions: UsefulnessStats
    sop_grounded_actions: UsefulnessStats = Field(default_factory=UsefulnessStats, description=(
        "action_useful ratings on inspections whose actions were SOP-grounded (RAG status GROUNDED)."))
    learning_status: dict[str, int]
    learning_loop: LearningLoopStatus = Field(default_factory=LearningLoopStatus)


class LearningCandidate(BaseModel):
    """One reviewed inspection as a dataset / evaluation CANDIDATE. Not a training sample until curated."""
    feedback_id: str
    inspection_id: str
    item_id: str | None
    order_id: str | None
    store_id: str
    sku_id: str
    inspection_stage: FulfillmentStage
    image_id: str | None
    image_sha256: str | None
    ai_decision: DecisionOutcome | None
    operator_decision: DecisionOutcome
    ai_defects: list[DefectType] | None
    operator_defects: list[DefectType] | None = Field(description="null when the operator gave no defect label.")
    ai_stage: FulfillmentStage | None
    operator_stage: FulfillmentStage | None
    decision_correct: bool | None
    defect_correct: bool | None
    stage_correct: bool | None
    rca_useful: bool | None
    action_useful: bool | None
    correction_layers: list[str]
    comment: str | None = Field(description="Operator free text: untrusted data.")
    model_versions: dict[str, str]
    learning_status: LearningStatus
    source: str = "operator_feedback"
    requires_curation: bool = True
    created_at: datetime
