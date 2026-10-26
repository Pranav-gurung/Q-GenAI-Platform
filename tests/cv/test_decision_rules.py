import pytest

from backend.core.config import Settings
from backend.domain.enums import DecisionOutcome, DecisionSource, DefectType, ModuleStatus, Severity
from backend.domain.fulfillment import FulfillmentStage
from backend.pipeline.context import InspectionInfo
from backend.pipeline.decision_rules import RulesDecisionEngine
from backend.schemas.results import BoundingBox, CVResult, Detection

INFO = InspectionInfo("INS-0", "DS-BLR-001", "SKU-1", FulfillmentStage.PACKING)


def cv(*dets):
    return CVResult(module="yolov8", module_version="v-test", status=ModuleStatus.COMPLETED,
                    detections=list(dets), classification="DEFECTS_DETECTED" if dets else "NO_DEFECTS_DETECTED")


def det(sev, conf, t=DefectType.DAMAGED_BOX):
    return Detection(defect_type=t, confidence=conf, severity=sev, bbox=BoundingBox(x_min=0, y_min=0, x_max=5, y_max=5))


def engine(**kw):
    return RulesDecisionEngine(Settings(_env_file=None, **kw))


@pytest.mark.parametrize("status", [ModuleStatus.NOT_IMPLEMENTED, ModuleStatus.UNAVAILABLE, ModuleStatus.FAILED])
def test_no_real_cv_never_decides(status):
    d = engine(decision_auto_pass=True).decide(CVResult(module="yolov8", status=status), INFO)
    assert d.status is ModuleStatus.SKIPPED and d.outcome is None


def test_no_detections_review_by_default_pass_when_enabled():
    d = engine().decide(cv(), INFO)
    assert d.outcome is DecisionOutcome.REVIEW and "DECISION_AUTO_PASS" in d.reason
    p = engine(decision_auto_pass=True).decide(cv(), INFO)
    assert p.outcome is DecisionOutcome.PASS and p.decision_source is DecisionSource.RULES
    assert p.confidence is None  # no fabricated confidence for "nothing found"


def test_high_severity_confident_detection_fails():
    d = engine().decide(cv(det(Severity.LOW, 0.9), det(Severity.HIGH, 0.7, DefectType.LEAKING_PACKAGE)), INFO)
    assert d.outcome is DecisionOutcome.FAIL
    assert d.severity is Severity.HIGH and d.confidence == pytest.approx(0.7)
    assert "LEAKING_PACKAGE" in d.reason and d.module_version == "1"


def test_low_confidence_or_low_severity_goes_to_review():
    assert engine().decide(cv(det(Severity.CRITICAL, 0.4)), INFO).outcome is DecisionOutcome.REVIEW
    assert engine().decide(cv(det(Severity.MEDIUM, 0.99)), INFO).outcome is DecisionOutcome.REVIEW


def test_thresholds_configurable():
    e = engine(decision_fail_min_severity="MEDIUM", decision_fail_min_confidence=0.3)
    assert e.decide(cv(det(Severity.MEDIUM, 0.31)), INFO).outcome is DecisionOutcome.FAIL
