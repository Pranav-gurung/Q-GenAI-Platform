"""On-demand RCA for an inspection: persisted evidence -> Part 3 analysis -> RCAService.

Not cached and not stored in a new table: every call reflects the current evidence (including operator
corrections). The snapshot produced during the pipeline run is stored in the existing inspection_results.rca
column. Each call may make one synchronous LLM request (bounded by LLM_TIMEOUT_SECONDS).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Feedback, Inspection
from backend.rca.service import RCAService
from backend.schemas.rca import OperatorNote, RootCauseAnalysis
from backend.services.fulfillment_service import analyze_inspection_trace


def operator_notes(session: Session, inspection_ids: list[str], stages: dict[str, object]) -> list[OperatorNote]:
    """Latest non-empty operator comment per inspection in the trace. Untrusted text, passed as data only."""
    latest: dict[str, Feedback] = {}
    for fb in session.scalars(select(Feedback).where(Feedback.inspection_id.in_(inspection_ids),
                                                     Feedback.comment.is_not(None))
                              .order_by(Feedback.created_at, Feedback.feedback_id)):
        latest[fb.inspection_id] = fb
    return [OperatorNote(inspection_id=i, stage=stages[i], note=fb.comment[:300])
            for i, fb in sorted(latest.items()) if fb.comment and fb.comment.strip()]


def generate_inspection_rca(session: Session, insp: Inspection, service: RCAService) -> RootCauseAnalysis:
    trace = analyze_inspection_trace(session, insp)
    stages = {o.inspection_id: o.stage for o in trace.observations}
    notes = operator_notes(session, list(stages), stages) if stages else []
    return service.analyze(trace, notes)
