"""Operator feedback: InspectionResult + human feedback -> FeedbackRecord.

Records are stored with learning_status=PENDING; nothing is trained and no rule changes here (Part 6 only adds
error analysis and candidate export, which a human reviews). Feedback on an inspection in REQUIRES_REVIEW resolves
it (-> COMPLETED with the operator's verdict as the final decision); the persisted AI result is never modified.

Part 6 review: before storing, the AI output the operator is reviewing is snapshotted (decision, CV defect types,
first observed stage, RCA / RAG status). A review of a layer the AI did not produce is rejected
(AI_RESULT_NOT_AVAILABLE), and "correct" / "incorrect" must agree with the submitted values (FEEDBACK_INCONSISTENT).
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.exceptions import ConflictError, ValidationFailedError
from backend.core.logging import log_pipeline
from backend.db.models import Feedback, Inspection, PipelineEvent
from backend.domain.enums import DecisionOutcome, LearningStatus, ModuleStatus
from backend.domain.fulfillment import FULFILLMENT_FLOW, FulfillmentStage
from backend.pipeline.states import FEEDBACK_STATES, PipelineState
from backend.pipeline.transitions import ensure_transition
from backend.schemas.common import new_id, utcnow
from backend.schemas.feedback import FeedbackCreate
from backend.schemas.fulfillment import FulfillmentTraceAnalysis
from backend.services.fulfillment_service import analyze_inspection_trace
from backend.services.inspection_service import InspectionService

logger = logging.getLogger(__name__)


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
        snapshot = ai_snapshot(self.session, insp)
        system_outcome = (snapshot["decision"] or {}).get("outcome")
        outcome, defect_types, stage = _apply_review(data, snapshot)

        now = utcnow()
        fb = Feedback(
            feedback_id=new_id("FB"), inspection_id=inspection_id, result_id=result.result_id if result else None,
            operator_id=data.operator_id, outcome=outcome, defect_types=defect_types,
            corrected_detections=[d.model_dump(mode="json") for d in data.corrected_detections],
            suspected_stage=stage, comment=data.comment, system_outcome=system_outcome,
            is_correction=system_outcome is not None and system_outcome != outcome.value,
            model_versions=dict(result.model_versions) if result else {},
            decision_correct=data.decision_correct, defect_correct=data.defect_correct,
            stage_correct=data.stage_correct, rca_useful=data.rca_useful, action_useful=data.action_useful,
            ai_defect_types=snapshot["defects"], ai_stage=snapshot["pipeline"]["first_observed_stage"]
            if snapshot["pipeline"] else None, ai_snapshot=snapshot,
            learning_status=LearningStatus.PENDING, created_at=now,
        )
        self.session.add(fb)

        if insp.status is PipelineState.REQUIRES_REVIEW:
            ensure_transition(insp.status, PipelineState.COMPLETED)
            insp.events.append(PipelineEvent(event_type="STATE_TRANSITION", from_state=insp.status,
                                             to_state=PipelineState.COMPLETED, created_at=now,
                                             message=f"Resolved by operator review ({outcome.value})"))
            insp.status = PipelineState.COMPLETED
            insp.decision = outcome
            insp.updated_at = now
        self.session.commit()
        log_pipeline(inspection_id, "FEEDBACK", "RECORDED",
                     message=f"feedback={fb.feedback_id} outcome={outcome.value} correction={fb.is_correction} "
                             f"review={_review_flags(fb)}")
        return fb

    def list_for(self, inspection_id: str) -> list[Feedback]:
        return list(self.session.scalars(select(Feedback).where(Feedback.inspection_id == inspection_id)
                                         .order_by(Feedback.created_at)))


# ------------------------------------------------------------------------------------------------ AI snapshot
def ai_snapshot(session: Session, insp: Inspection) -> dict:
    """What the AI showed for this inspection, taken from persisted results (never recomputed by a model)."""
    result = insp.result
    decision = (result.decision or {}) if result else {}
    cv = (result.cv or {}) if result else {}
    rca = (result.rca or {}) if result else {}
    action = (result.action or {}) if result else {}
    done = ModuleStatus.COMPLETED.value
    snap: dict = {
        "decision": {"outcome": decision.get("outcome"), "severity": decision.get("severity"),
                     "module_version": decision.get("module_version")} if decision.get("status") == done else None,
        "defects": sorted({d["defect_type"] for d in cv.get("detections", [])}) if cv.get("status") == done else None,
        "pipeline": _pipeline_snapshot(session, insp),
        "rca": None, "rag": None,
        "actions": {"count": len(action.get("actions", [])),
                    "sop_backed": sum(1 for a in action.get("actions", []) if a.get("sop_reference"))}
        if action.get("status") == done else None,
    }
    analysis = rca.get("root_cause_analysis") if rca.get("status") == done else None
    if analysis:
        g = analysis.get("generation") or {}
        snap["rca"] = {"status": analysis.get("status"), "provider": g.get("provider"), "model": g.get("model"),
                       "prompt_version": g.get("prompt_version")}
        rag = analysis.get("rag")
        if rag:
            snap["rag"] = {"status": rag.get("status"), "grounded_actions": len(rag.get("grounded_actions", [])),
                           "sources": [s["chunk_id"] for s in rag.get("retrieved_sources", [])]}
    return snap


def _pipeline_snapshot(session: Session, insp: Inspection) -> dict | None:
    """Stage attribution shown for this inspection: the pipeline run's analysis, else the on-request analysis
    (GET /pipeline). first_observed_stage = earliest first observed stage of the defects in this inspection."""
    stored = (insp.result.pipeline_analysis or {}) if insp.result else {}
    try:
        if stored.get("status") == ModuleStatus.COMPLETED.value and stored.get("trace_analysis"):
            source, trace = "PIPELINE_RESULT", FulfillmentTraceAnalysis.model_validate(stored["trace_analysis"])
        else:
            source, trace = "TRACE_ANALYSIS", analyze_inspection_trace(session, insp)
    except Exception:  # never block feedback on the analysis
        logger.exception("stage snapshot failed for %s", insp.inspection_id)
        return None
    here = [a for a in trace.defect_analyses if a.current_stage_status is not None
            and a.current_stage_status.value != "NOT_PRESENT"] or trace.defect_analyses
    if not here:
        return None
    first = min((a.first_observed_stage for a in here), key=FULFILLMENT_FLOW.index)
    return {"source": source, "analyzer_version": trace.analyzer_version, "first_observed_stage": first.value,
            "defects": [{"defect_type": a.defect_type.value, "first_observed_stage": a.first_observed_stage.value,
                         "last_known_good_stage": a.last_known_good_stage.value if a.last_known_good_stage else None,
                         "classification": a.classification.value, "evidence_level": a.evidence_level.value}
                        for a in here]}


# ------------------------------------------------------------------------------------------------ review validation
def _invalid(code: str, message: str, field: str) -> ValidationFailedError:
    return ValidationFailedError(message, code=code, details={"field": field})


def _apply_review(data: FeedbackCreate, snap: dict) -> tuple[DecisionOutcome, list[str], FulfillmentStage | None]:
    """Resolve the operator values from the review flags and check they agree with the AI snapshot."""
    outcome = data.outcome
    ai_decision = (snap["decision"] or {}).get("outcome")
    if data.decision_correct is not None:
        if ai_decision is None:
            raise _invalid("AI_RESULT_NOT_AVAILABLE", "There is no automated decision to review", "decision_correct")
        if data.decision_correct:
            if outcome is not None and outcome.value != ai_decision:
                raise _invalid("FEEDBACK_INCONSISTENT", "decision_correct is true but outcome differs from the AI "
                                                         "decision", "outcome")
            outcome = DecisionOutcome(ai_decision)
        elif outcome.value == ai_decision:
            raise _invalid("FEEDBACK_INCONSISTENT", "decision_correct is false but outcome equals the AI decision "
                                                     "(give the corrected decision)", "outcome")

    defects = sorted({d.value for d in data.defect_types})
    if data.defect_correct is not None:
        ai = snap["defects"]
        if ai is None:
            raise _invalid("AI_RESULT_NOT_AVAILABLE", "There is no completed CV result to review", "defect_correct")
        if data.defect_correct:
            if defects and defects != ai:
                raise _invalid("FEEDBACK_INCONSISTENT", "defect_correct is true but defect_types differ from the AI "
                                                         "defects", "defect_types")
            defects = list(ai)
        else:
            if defects == ai:
                raise _invalid("FEEDBACK_INCONSISTENT", "defect_correct is false but defect_types equal the AI "
                                                         "defects (give the corrected defect types)", "defect_types")
            if not defects and outcome is not DecisionOutcome.PASS:
                raise _invalid("FEEDBACK_INCONSISTENT", "defect_types (the corrected defects) is required when "
                                                         "defect_correct is false", "defect_types")

    stage = data.suspected_stage
    if data.stage_correct is not None:
        ai_stage = (snap["pipeline"] or {}).get("first_observed_stage")
        if ai_stage is None:
            raise _invalid("AI_RESULT_NOT_AVAILABLE", "There is no stage attribution to review", "stage_correct")
        if data.stage_correct:
            if stage is not None and stage.value != ai_stage:
                raise _invalid("FEEDBACK_INCONSISTENT", "stage_correct is true but suspected_stage differs from the "
                                                         "attributed stage", "suspected_stage")
            stage = FulfillmentStage(ai_stage)
        elif stage.value == ai_stage:
            raise _invalid("FEEDBACK_INCONSISTENT", "stage_correct is false but suspected_stage equals the attributed "
                                                     "stage (give the corrected stage)", "suspected_stage")
    return outcome, defects, stage


def _review_flags(fb: Feedback) -> str:
    names = ("decision_correct", "defect_correct", "stage_correct", "rca_useful", "action_useful")
    return ",".join(f"{n}={getattr(fb, n)}" for n in names if getattr(fb, n) is not None) or "none"
