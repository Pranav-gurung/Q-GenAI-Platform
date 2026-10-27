from backend.domain.enums import DecisionOutcome, DecisionSource, DefectType, ModuleStatus, Severity
from backend.fulfillment.observations import build_observation, trace_key_for
from backend.schemas.fulfillment import EvidenceSource, ObservationStatus, TraceKeyType
from backend.schemas.results import BoundingBox, CVResult, Detection, QualityDecision
from tests.fulfillment.helpers import K, T0

BOX = BoundingBox(x_min=0, y_min=0, x_max=5, y_max=5)
CV = CVResult(module="yolov8", module_version="v1", status=ModuleStatus.COMPLETED, detections=[
    Detection(defect_type=DefectType.LEAKING_PACKAGE, confidence=0.6, severity=Severity.HIGH, bbox=BOX),
    Detection(defect_type=DefectType.LEAKING_PACKAGE, confidence=0.9, severity=Severity.CRITICAL, bbox=BOX),
    Detection(defect_type=DefectType.MISSING_LABEL, confidence=0.7, severity=Severity.MEDIUM, bbox=BOX)])


def decision(outcome):
    return QualityDecision(module="rules", module_version="1", status=ModuleStatus.COMPLETED, outcome=outcome,
                           decision_source=DecisionSource.RULES)


def build(**kw):
    base = dict(inspection_id="INS-1", stage=K, observed_at=T0, store_id="S", sku_id="K", item_id="U", order_id=None,
                decision=None, cv=None)
    return build_observation(**{**base, **kw})


def test_real_decision_with_aggregated_cv_defects():
    o = build(decision=decision(DecisionOutcome.FAIL), cv=CV)
    assert o.status is ObservationStatus.FAIL and o.decision_source is EvidenceSource.MODEL and o.model_version == "v1"
    leak = next(d for d in o.defects if d.defect_type is DefectType.LEAKING_PACKAGE)
    assert leak.confidence == 0.9 and leak.severity is Severity.CRITICAL and len(o.defects) == 2


def test_no_decision_is_review_never_pass():
    for cv in (None, CVResult.unavailable("yolov8", "MODEL_NOT_FOUND"), CVResult.not_implemented("stub", "x")):
        o = build(decision=QualityDecision.skipped("rules", "no cv"), cv=cv)
        assert o.status is ObservationStatus.REVIEW and o.decision_source is EvidenceSource.NO_DECISION
        assert o.defects == [] and o.decision is None and o.model_version is None


def test_operator_feedback_overrides():
    o = build(decision=decision(DecisionOutcome.PASS), cv=CV, operator_outcome=DecisionOutcome.FAIL,
              operator_defects=[DefectType.CRUSHED_PACKAGE])
    assert o.status is ObservationStatus.FAIL and o.decision_source is EvidenceSource.OPERATOR
    assert [d.defect_type for d in o.defects] == [DefectType.CRUSHED_PACKAGE] and o.defects[0].confidence is None
    passed = build(decision=decision(DecisionOutcome.FAIL), cv=CV, operator_outcome=DecisionOutcome.PASS)
    assert passed.status is ObservationStatus.PASS and passed.defects == []
    untyped = build(decision=decision(DecisionOutcome.REVIEW), cv=CV, operator_outcome=DecisionOutcome.FAIL)
    assert {d.source for d in untyped.defects} == {EvidenceSource.MODEL} and len(untyped.defects) == 2


def test_trace_keys():
    assert trace_key_for("S", "K", "U", "O").key_type is TraceKeyType.ITEM
    assert trace_key_for("S", "K", None, "O").key_type is TraceKeyType.ORDER
    assert trace_key_for("S", "K", None, None).key_type is TraceKeyType.NONE


def test_internal_fields_not_serialised():
    dumped = build(image_sha256="abc").model_dump(mode="json")
    assert not {"image_sha256", "store_id", "sku_id", "item_id", "order_id"} & set(dumped)


def test_operator_verdict_keeps_label_when_a_later_review_omits_types():
    """Part 8: a later review that restates FAIL without defect types must not erase the earlier operator label."""
    from types import SimpleNamespace as F

    from backend.fulfillment.observations import operator_verdict
    leak = ["LEAKING_PACKAGE"]
    assert operator_verdict([]) is None
    assert operator_verdict([F(outcome="FAIL", defect_types=leak), F(outcome="FAIL", defect_types=[])]) == (
        DecisionOutcome.FAIL, [DefectType.LEAKING_PACKAGE])
    assert operator_verdict([F(outcome="FAIL", defect_types=leak), F(outcome="REVIEW", defect_types=None)])[1] == [
        DefectType.LEAKING_PACKAGE]
    assert operator_verdict([F(outcome="FAIL", defect_types=leak), F(outcome="FAIL", defect_types=["TORN_PACKAGE"])])[
        1] == [DefectType.TORN_PACKAGE]  # an explicit new label replaces the old one
    assert operator_verdict([F(outcome="FAIL", defect_types=leak), F(outcome="PASS", defect_types=[])]) == (
        DecisionOutcome.PASS, [])
    assert operator_verdict([F(outcome="FAIL", defect_types=leak), F(outcome="PASS", defect_types=[]),
                             F(outcome="FAIL", defect_types=[])]) == (DecisionOutcome.FAIL, [])  # PASS cleared it
