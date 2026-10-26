"""Normalise existing inspection results into PipelineObservations (no re-inference).

Precedence for one inspection:
  1. Operator feedback (latest) - status = operator outcome; defects = operator's defect types. If the operator
     gave no types: PASS -> no defects, FAIL/REVIEW -> the CV detections (source MODEL).
  2. A real decision (rules engine) - status = PASS/FAIL/REVIEW, defects = CV detection types.
  3. No usable decision (quality rejected, CV unavailable, failed run, stub) - status REVIEW: an inspection
     that could not decide is NEVER treated as a clean PASS. Real CV detections, if any, are kept.
"""
from __future__ import annotations

from datetime import datetime

from backend.domain.enums import DecisionOutcome, DefectType
from backend.domain.fulfillment import FulfillmentStage
from backend.domain.severity import max_severity
from backend.schemas.fulfillment import (EvidenceSource, ObservationStatus, ObservedDefect, PipelineObservation,
                                         TraceKey, TraceKeyType)
from backend.schemas.results import CVResult, QualityDecision


def trace_key_for(store_id: str, sku_id: str, item_id: str | None, order_id: str | None) -> TraceKey:
    """item_id (physical unit / handling unit) links all stages; order_id only exists from PICKING onwards."""
    if item_id:
        return TraceKey(key_type=TraceKeyType.ITEM, store_id=store_id, sku_id=sku_id, item_id=item_id)
    if order_id:
        return TraceKey(key_type=TraceKeyType.ORDER, store_id=store_id, sku_id=sku_id, order_id=order_id)
    return TraceKey(key_type=TraceKeyType.NONE, store_id=store_id, sku_id=sku_id)


def _cv_defects(cv: CVResult | None) -> list[ObservedDefect]:
    if cv is None or not cv.is_real:
        return []
    by_type: dict[DefectType, list] = {}
    for d in cv.detections:
        by_type.setdefault(d.defect_type, []).append(d)
    return [ObservedDefect(defect_type=t, severity=max_severity(d.severity for d in ds),
                           confidence=max(d.confidence for d in ds), source=EvidenceSource.MODEL)
            for t, ds in sorted(by_type.items(), key=lambda kv: kv[0].value)]


def build_observation(*, inspection_id: str, stage: FulfillmentStage, observed_at: datetime, store_id: str,
                      sku_id: str, item_id: str | None, order_id: str | None, decision: QualityDecision | None,
                      cv: CVResult | None, operator_outcome: DecisionOutcome | None = None,
                      operator_defects: list[DefectType] | None = None, image_id: str | None = None,
                      image_sha256: str | None = None) -> PipelineObservation:
    cv_defects = _cv_defects(cv)
    if operator_outcome is not None:
        status, source, final = ObservationStatus(operator_outcome.value), EvidenceSource.OPERATOR, operator_outcome
        if operator_defects:
            defects = [ObservedDefect(defect_type=t, source=EvidenceSource.OPERATOR)
                       for t in sorted(set(operator_defects), key=lambda t: t.value)]
        else:
            defects = [] if operator_outcome is DecisionOutcome.PASS else cv_defects
    elif decision is not None and decision.is_real and decision.outcome is not None:
        status, source, final = ObservationStatus(decision.outcome.value), EvidenceSource.MODEL, decision.outcome
        defects = cv_defects
    else:
        status, source, final, defects = ObservationStatus.REVIEW, EvidenceSource.NO_DECISION, None, cv_defects
    return PipelineObservation(
        inspection_id=inspection_id, stage=stage, observed_at=observed_at, status=status, decision=final,
        decision_source=source, defects=defects,
        model_version=cv.module_version if cv is not None and cv.is_real else None, image_id=image_id,
        store_id=store_id, sku_id=sku_id, item_id=item_id, order_id=order_id, image_sha256=image_sha256)
