"""Deterministic decision engine: CVResult -> QualityDecision (PASS / FAIL / REVIEW).

Computer vision only reports visual evidence (what, where, how confident, how severe).
This module applies explicit, configurable business rules to that evidence:

  1. CV did not actually run (stub / UNAVAILABLE / FAILED)  -> SKIPPED (no decision; pipeline routes
     the inspection to human review). Never PASS without real inference.
  2. Any detection with severity >= DECISION_FAIL_MIN_SEVERITY and
     confidence >= DECISION_FAIL_MIN_CONFIDENCE              -> FAIL
  3. Other detections (lower severity or lower confidence)    -> REVIEW
  4. No detections:  DECISION_AUTO_PASS=true                  -> PASS
                     DECISION_AUTO_PASS=false (default)       -> REVIEW
     Auto-pass is off by default because the current model only covers some defect classes
     (see ml/cv/taxonomy.yaml); "nothing detected" is not proof that nothing is wrong.
"""
from __future__ import annotations

from backend.core.config import Settings
from backend.domain.enums import DecisionOutcome, DecisionSource, ModuleStatus, Severity
from backend.domain.severity import max_severity, severity_rank
from backend.pipeline.context import InspectionInfo
from backend.pipeline.interfaces import DecisionEngine
from backend.schemas.results import CVResult, QualityDecision

RULES_VERSION = "1"


class RulesDecisionEngine(DecisionEngine):
    name = "rules"

    def __init__(self, settings: Settings) -> None:
        self.fail_min_severity = Severity(settings.decision_fail_min_severity)
        self.fail_min_confidence = settings.decision_fail_min_confidence
        self.auto_pass = settings.decision_auto_pass

    def health(self) -> dict:
        return {"available": True, "version": RULES_VERSION,
                "fail_min_severity": self.fail_min_severity.value,
                "fail_min_confidence": self.fail_min_confidence, "auto_pass": self.auto_pass}

    def _decision(self, outcome: DecisionOutcome, reason: str, *, confidence=None, severity=None) -> QualityDecision:
        return QualityDecision(module=self.name, module_version=RULES_VERSION, status=ModuleStatus.COMPLETED,
                               outcome=outcome, reason=reason[:500], confidence=confidence, severity=severity,
                               decision_source=DecisionSource.RULES)

    def decide(self, cv_result: CVResult, inspection: InspectionInfo) -> QualityDecision:
        if not cv_result.is_real:
            return QualityDecision.skipped(
                self.name, f"No computer vision result to decide on (CV status: {cv_result.status.value})")

        detections = cv_result.detections
        source = f"{cv_result.module}:{cv_result.module_version}"
        if not detections:
            if self.auto_pass:
                return self._decision(DecisionOutcome.PASS, f"No defects detected by {source}")
            return self._decision(DecisionOutcome.REVIEW, f"No defects detected by {source}; automatic PASS is "
                                                          "disabled (DECISION_AUTO_PASS=false), human review required")

        overall = max_severity(d.severity for d in detections)
        failing = [d for d in detections if d.severity is not None
                   and severity_rank(d.severity) >= severity_rank(self.fail_min_severity)
                   and d.confidence >= self.fail_min_confidence]
        if failing:
            top = max(failing, key=lambda d: d.confidence)
            summary = ", ".join(sorted({f"{d.defect_type.value} ({d.severity.value})" for d in failing}))
            return self._decision(DecisionOutcome.FAIL, f"{len(failing)} defect(s) at or above "
                                  f"{self.fail_min_severity.value} severity with confidence >= "
                                  f"{self.fail_min_confidence:g}: {summary}",
                                  confidence=top.confidence, severity=overall)
        top = max(detections, key=lambda d: d.confidence)
        return self._decision(DecisionOutcome.REVIEW, f"{len(detections)} defect(s) detected below the automatic "
                              f"FAIL criteria (highest: {top.defect_type.value}, {top.severity.value if top.severity else '-'}, "
                              f"confidence {top.confidence:.2f}); human review required",
                              confidence=top.confidence, severity=overall)
