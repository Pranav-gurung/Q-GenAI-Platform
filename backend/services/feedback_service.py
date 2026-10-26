"""Operator feedback: InspectionResult + human feedback -> FeedbackRecord.

Records are stored with learning_status=PENDING for the future active-learning loop (Part 6);
nothing is trained here. Feedback on an inspection in REQUIRES_REVIEW resolves it
(-> COMPLETED with the operator's verdict as the final decision).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.exceptions import ConflictError
from backend.core.logging import log_pipeline
from backend.db.models import Feedback, PipelineEvent
from backend.domain.enums import LearningStatus, ModuleStatus
from backend.pipeline.states import FEEDBACK_STATES, PipelineState
from backend.pipeline.transitions import ensure_transition
from backend.schemas.common import new_id, utcnow
from backend.schemas.feedback import FeedbackCreate
from backend.services.inspection_service import InspectionService


class FeedbackService:
    def __init__(self, session: Session, inspections: InspectionService) -> None:
        self.session = session
        self.inspections = inspections

    def submit(self, inspection_id: str, data: FeedbackCreate) -> Feedback:
        insp = self.inspections.get(inspection_id)
        if insp.status not in FEEDBACK_STATES:
            raise ConflictError(f"Feedback is not accepted while inspection is {insp.status.value}",
                                code="FEEDBACK_NOT_ALLOWED",
                                details={"status": insp.status.value,
                                         "allowed_statuses": sorted(s.value for s in FEEDBACK_STATES)})

        result = insp.result
        decision = (result.decision or {}) if result else {}
        system_outcome = decision.get("outcome") if decision.get("status") == ModuleStatus.COMPLETED.value else None

        now = utcnow()
        fb = Feedback(
            feedback_id=new_id("FB"), inspection_id=inspection_id, result_id=result.result_id if result else None,
            operator_id=data.operator_id, outcome=data.outcome,
            defect_types=[d.value for d in data.defect_types],
            corrected_detections=[d.model_dump(mode="json") for d in data.corrected_detections],
            suspected_stage=data.suspected_stage, comment=data.comment, system_outcome=system_outcome,
            is_correction=system_outcome is not None and system_outcome != data.outcome.value,
            model_versions=dict(result.model_versions) if result else {},
            learning_status=LearningStatus.PENDING, created_at=now,
        )
        self.session.add(fb)

        if insp.status is PipelineState.REQUIRES_REVIEW:
            ensure_transition(insp.status, PipelineState.COMPLETED)
            insp.events.append(PipelineEvent(event_type="STATE_TRANSITION", from_state=insp.status,
                                             to_state=PipelineState.COMPLETED, created_at=now,
                                             message=f"Resolved by operator review ({data.outcome.value})"))
            insp.status = PipelineState.COMPLETED
            insp.decision = data.outcome
            insp.updated_at = now
        self.session.commit()
        log_pipeline(inspection_id, "FEEDBACK", "RECORDED",
                     message=f"feedback={fb.feedback_id} outcome={data.outcome.value} correction={fb.is_correction}")
        return fb

    def list_for(self, inspection_id: str) -> list[Feedback]:
        return list(self.session.scalars(select(Feedback).where(Feedback.inspection_id == inspection_id)
                                         .order_by(Feedback.created_at)))
