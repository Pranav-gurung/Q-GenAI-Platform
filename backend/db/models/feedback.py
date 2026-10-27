"""Operator feedback (labels for the future active-learning loop).

Part 6 adds a structured review of each AI layer (decision / defect / stage attribution correct?, RCA / action
useful?) plus `ai_snapshot`: what the AI showed at review time. The AI result itself (inspection_results) is never
modified by feedback; the AI value and the operator value are stored side by side. One row per submission;
later submissions never overwrite earlier ones.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base, JSONType, UTCDateTime, str_enum
from backend.domain.enums import DecisionOutcome, LearningStatus
from backend.domain.fulfillment import FulfillmentStage
from backend.schemas.common import utcnow


class Feedback(Base):
    __tablename__ = "feedback"

    feedback_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.inspection_id", ondelete="CASCADE"),
                                               index=True)
    result_id: Mapped[str | None] = mapped_column(ForeignKey("inspection_results.result_id", ondelete="SET NULL"))
    operator_id: Mapped[str] = mapped_column(String(64), index=True)
    outcome: Mapped[DecisionOutcome] = mapped_column(str_enum(DecisionOutcome))
    defect_types: Mapped[list] = mapped_column(JSONType, default=list)
    corrected_detections: Mapped[list] = mapped_column(JSONType, default=list)
    suspected_stage: Mapped[FulfillmentStage | None] = mapped_column(str_enum(FulfillmentStage))
    comment: Mapped[str | None] = mapped_column(Text)
    system_outcome: Mapped[DecisionOutcome | None] = mapped_column(str_enum(DecisionOutcome))
    is_correction: Mapped[bool] = mapped_column(Boolean, default=False)
    model_versions: Mapped[dict] = mapped_column(JSONType, default=dict)
    # Part 6: structured review (null = not reviewed). Corrections reuse the fields above: `outcome` (decision),
    # `defect_types` (defects) and `suspected_stage` (stage attribution).
    decision_correct: Mapped[bool | None] = mapped_column(Boolean)
    defect_correct: Mapped[bool | None] = mapped_column(Boolean)
    stage_correct: Mapped[bool | None] = mapped_column(Boolean)
    rca_useful: Mapped[bool | None] = mapped_column(Boolean)
    action_useful: Mapped[bool | None] = mapped_column(Boolean)
    ai_defect_types: Mapped[list | None] = mapped_column(JSONType)          # null: no completed CV result
    ai_stage: Mapped[FulfillmentStage | None] = mapped_column(str_enum(FulfillmentStage), index=True)
    ai_snapshot: Mapped[dict] = mapped_column(JSONType, default=dict)
    learning_status: Mapped[LearningStatus] = mapped_column(str_enum(LearningStatus), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)

    # read-only link for item / order / store context (trace key); feedback never edits the inspection's AI result
    inspection: Mapped["Inspection"] = relationship(viewonly=True)  # noqa: F821
