"""Load the persisted observation history of an item / order trace and analyse it.

The analysis is always RECONSTRUCTED from persisted inspections, results and operator feedback (no separate
analysis table): the same input always gives the same output, and corrections (feedback) are reflected
immediately. The snapshot taken during the pipeline run is kept in inspection_results.pipeline_analysis.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.core.logging import log_pipeline
from backend.db.models import Feedback, Inspection
from backend.domain.enums import DecisionOutcome, DefectType
from backend.fulfillment.analyzer import analyze_trace
from backend.fulfillment.engine import log_analysis
from backend.fulfillment.observations import build_observation, trace_key_for
from backend.pipeline.states import PipelineState
from backend.schemas.fulfillment import FulfillmentTraceAnalysis, PipelineObservation, TraceKeyType
from backend.schemas.results import CVResult, QualityDecision

# Inspections that produced an observation. In-flight states (CREATED ... ACTION_GENERATED) are not evidence yet.
OBSERVED_STATES = (PipelineState.COMPLETED, PipelineState.REQUIRES_REVIEW, PipelineState.FAILED)
MAX_TRACE_OBSERVATIONS = 500  # defensive bound; a real item has a handful of inspections


def _trace_query(insp: Inspection):
    key = trace_key_for(insp.store_id, insp.sku_id, insp.item_id, insp.order_id)
    q = select(Inspection).where(Inspection.store_id == insp.store_id, Inspection.sku_id == insp.sku_id,
                                 Inspection.status.in_(OBSERVED_STATES))
    if key.key_type is TraceKeyType.ITEM:
        q = q.where(Inspection.item_id == key.item_id)
    elif key.key_type is TraceKeyType.ORDER:
        q = q.where(Inspection.order_id == key.order_id)
    else:
        q = q.where(Inspection.inspection_id == insp.inspection_id)  # unlinkable: only itself
    return key, q.options(selectinload(Inspection.images), selectinload(Inspection.result)) \
        .order_by(Inspection.created_at, Inspection.inspection_id).limit(MAX_TRACE_OBSERVATIONS)


def observation_from_row(row: Inspection, feedback: Feedback | None) -> PipelineObservation:
    result = row.result
    image = row.current_image
    return build_observation(
        inspection_id=row.inspection_id, stage=row.stage,
        observed_at=image.created_at if image else row.created_at,
        store_id=row.store_id, sku_id=row.sku_id, item_id=row.item_id, order_id=row.order_id,
        decision=QualityDecision.model_validate(result.decision) if result and result.decision else None,
        cv=CVResult.model_validate(result.cv) if result and result.cv else None,
        operator_outcome=DecisionOutcome(feedback.outcome) if feedback else None,
        operator_defects=[DefectType(t) for t in (feedback.defect_types or [])] if feedback else None,
        image_id=image.image_id if image else None, image_sha256=image.sha256 if image else None)


def load_trace_observations(session: Session, insp: Inspection, *, exclude_current: bool) -> list[PipelineObservation]:
    _, query = _trace_query(insp)
    rows = [r for r in session.scalars(query) if not (exclude_current and r.inspection_id == insp.inspection_id)]
    if not rows:
        return []
    latest_feedback: dict[str, Feedback] = {}
    for fb in session.scalars(select(Feedback).where(Feedback.inspection_id.in_([r.inspection_id for r in rows]))
                              .order_by(Feedback.created_at, Feedback.feedback_id)):
        latest_feedback[fb.inspection_id] = fb  # ordered ascending -> last one wins
    return [observation_from_row(r, latest_feedback.get(r.inspection_id)) for r in rows]


def analyze_inspection_trace(session: Session, insp: Inspection) -> FulfillmentTraceAnalysis:
    """GET /api/inspections/{id}/pipeline: the item/order history this inspection belongs to, analysed now."""
    key = trace_key_for(insp.store_id, insp.sku_id, insp.item_id, insp.order_id)
    log_pipeline(insp.inspection_id, "PIPELINE_ANALYSIS", "STARTED", message=f"trace={key.key_type.value} (on request)")
    observations = load_trace_observations(session, insp, exclude_current=False)
    log_pipeline(insp.inspection_id, "PIPELINE_ANALYSIS", "OBSERVATIONS_LOADED", message=f"observations={len(observations)}")
    analysis = analyze_trace(key, observations, current_inspection_id=insp.inspection_id, current_stage=insp.stage)
    log_analysis(insp.inspection_id, analysis)
    return analysis
