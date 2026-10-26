"""Deterministic fixtures for fulfillment-trace tests (synthetic observations, test-only)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend.domain.enums import DecisionOutcome, DefectType, Severity
from backend.domain.fulfillment import FulfillmentStage as St
from backend.schemas.fulfillment import EvidenceSource, ObservationStatus, ObservedDefect, PipelineObservation, TraceKey, TraceKeyType

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
TRACE = TraceKey(key_type=TraceKeyType.ITEM, store_id="DS-BLR-001", sku_id="SKU-MILK-500ML", item_id="UNIT-1")
R, S, P, K, D, V = St.RECEIVING, St.STORAGE, St.PICKING, St.PACKING, St.DISPATCH, St.DELIVERY
CRUSHED, LEAK, DENT, LABEL = (DefectType.CRUSHED_PACKAGE, DefectType.LEAKING_PACKAGE, DefectType.DENTED_CONTAINER,
                              DefectType.MISSING_LABEL)

_ORDER = {s: i for i, s in enumerate([R, S, P, K, D, V])}


def obs(stage, status="PASS", defects=(), *, minute=None, iid=None, source=EvidenceSource.MODEL, sha=None,
        item="UNIT-1", order=None, sku="SKU-MILK-500ML") -> PipelineObservation:
    status = ObservationStatus(status)
    minute = _ORDER[stage] * 10 if minute is None else minute
    return PipelineObservation(
        inspection_id=iid or f"INS-{stage.value[:4]}{minute:04d}", stage=stage, observed_at=T0 + timedelta(minutes=minute),
        status=status, decision=None if status is ObservationStatus.REVIEW and source is EvidenceSource.NO_DECISION
        else DecisionOutcome(status.value), decision_source=source,
        defects=[ObservedDefect(defect_type=d, severity=Severity.HIGH, confidence=0.9 if source is EvidenceSource.MODEL else None,
                                source=source) for d in defects],
        store_id="DS-BLR-001", sku_id=sku, item_id=item, order_id=order, image_sha256=sha)
