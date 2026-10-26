"""Tests the orchestration itself, not AI.

The production stubs are used for the default flow. To exercise routing rules (PASS, REVIEW,
quality rejection, failures) the tests use small TEST DOUBLES defined here; they exist only
in tests and are never registered in the application.
"""
from dataclasses import replace

import pytest

from backend.core.config import Settings
from backend.domain.enums import DecisionOutcome, DecisionSource, ModuleStatus
from backend.domain.fulfillment import FulfillmentStage
from backend.pipeline.context import ImageInput, InspectionInfo, PipelineContext
from backend.pipeline.interfaces import CVEngine, DecisionEngine, ImageQualityGate
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.registry import build_modules
from backend.pipeline.states import PipelineState as S
from backend.schemas.common import utcnow
from backend.schemas.image import ImageMetadata
from backend.schemas.results import CVResult, ImageQualityResult, QualityDecision

STUBS = build_modules(Settings(_env_file=None))


def make_ctx(state=S.IMAGE_RECEIVED, with_image=True) -> PipelineContext:
    meta = ImageMetadata(image_id="IMG-0", inspection_id="INS-0", file_name="a.png", mime_type="image/png",
                         size_bytes=10, width=8, height=8, sha256="0" * 64, storage_backend="local",
                         storage_reference="INS-0/IMG-0.png", created_at=utcnow())
    return PipelineContext(
        inspection=InspectionInfo("INS-0", "DS-BLR-001", "SKU-MILK-500ML", FulfillmentStage.PACKING),
        state=state, image=ImageInput(meta, lambda: b"") if with_image else None)


def transitions(ctx):
    return [h.to_state for h in ctx.history if h.event_type == "STATE_TRANSITION"]


# ------------------------------------------------------------------ test doubles
class FixedDecision(DecisionEngine):
    name = "test-double"

    def __init__(self, outcome):
        self.outcome = outcome

    def decide(self, cv_result, inspection):
        return QualityDecision(module=self.name, module_version="t1", status=ModuleStatus.COMPLETED,
                               outcome=self.outcome, decision_source=DecisionSource.RULES, reason="test")


class RejectingQualityGate(ImageQualityGate):
    name = "test-double"

    def assess(self, image):
        return ImageQualityResult(module=self.name, module_version="t1", status=ModuleStatus.COMPLETED,
                                  acceptable=False, issues=["BLUR"])


class ExplodingCV(CVEngine):
    name = "exploding"

    def detect(self, image):
        raise RuntimeError("GPU on fire: secret internal detail")


class RaisesNotImplementedCV(CVEngine):
    name = "todo"

    def detect(self, image):
        raise NotImplementedError


class WrongTypeCV(CVEngine):
    name = "wrong"

    def detect(self, image):
        return {"detections": []}


# ------------------------------------------------------------------ tests
def test_stub_pipeline_walks_every_stage_and_requires_review():
    ctx = PipelineOrchestrator(STUBS).run(make_ctx())
    assert transitions(ctx) == [S.QUALITY_CHECK, S.CV_PROCESSING, S.DECISION, S.PIPELINE_ANALYSIS,
                                S.RCA_PROCESSING, S.ACTION_GENERATED, S.REQUIRES_REVIEW]
    assert ctx.state is S.REQUIRES_REVIEW
    assert ctx.history[-1].error_code == "NO_AUTOMATED_DECISION"
    # every module ran and honestly reported that it produced nothing
    statuses = {h.to_state: h.module_status for h in ctx.history if h.event_type == "MODULE_RESULT"}
    assert statuses[S.CV_PROCESSING] is ModuleStatus.NOT_IMPLEMENTED
    assert statuses[S.DECISION] is ModuleStatus.SKIPPED  # nothing to decide on without CV
    assert ctx.final_decision is None
    assert ctx.model_versions() == {}
    assert ctx.cv.detections == [] and ctx.cv.overall_confidence is None


def test_pass_decision_completes_without_rca():
    ctx = PipelineOrchestrator(replace(STUBS, decision_engine=FixedDecision(DecisionOutcome.PASS))).run(make_ctx())
    assert ctx.state is S.COMPLETED
    assert transitions(ctx)[-2:] == [S.DECISION, S.COMPLETED]
    assert ctx.rca is None
    assert ctx.final_decision is DecisionOutcome.PASS
    assert ctx.model_versions() == {"decision": "test-double:t1"}


def test_review_decision_requires_review():
    ctx = PipelineOrchestrator(replace(STUBS, decision_engine=FixedDecision(DecisionOutcome.REVIEW))).run(make_ctx())
    assert ctx.state is S.REQUIRES_REVIEW
    assert ctx.history[-1].error_code == "DECISION_REVIEW"


def test_fail_decision_without_action_requires_review():
    ctx = PipelineOrchestrator(replace(STUBS, decision_engine=FixedDecision(DecisionOutcome.FAIL))).run(make_ctx())
    assert ctx.state is S.REQUIRES_REVIEW
    assert ctx.history[-1].error_code == "NO_CORRECTIVE_ACTION"
    assert ctx.final_decision is DecisionOutcome.FAIL


def test_quality_rejection_stops_pipeline():
    ctx = PipelineOrchestrator(replace(STUBS, quality_gate=RejectingQualityGate())).run(make_ctx())
    assert transitions(ctx) == [S.QUALITY_CHECK, S.REQUIRES_REVIEW]
    assert ctx.cv is None


def test_module_exception_fails_pipeline_without_leaking_details():
    ctx = PipelineOrchestrator(replace(STUBS, cv_engine=ExplodingCV())).run(make_ctx())
    assert ctx.state is S.FAILED
    assert ctx.errors[-1].code == "MODULE_ERROR" and ctx.errors[-1].stage is S.CV_PROCESSING
    assert ctx.cv.status is ModuleStatus.FAILED
    assert "GPU on fire" not in ctx.errors[-1].message and "GPU on fire" not in (ctx.cv.message or "")


def test_not_implemented_error_is_recorded_not_failed():
    ctx = PipelineOrchestrator(replace(STUBS, cv_engine=RaisesNotImplementedCV())).run(make_ctx())
    assert ctx.cv.status is ModuleStatus.NOT_IMPLEMENTED
    assert ctx.state is S.REQUIRES_REVIEW


def test_contract_violation_fails_pipeline():
    ctx = PipelineOrchestrator(replace(STUBS, cv_engine=WrongTypeCV())).run(make_ctx())
    assert ctx.state is S.FAILED
    assert ctx.errors[-1].code == "MODULE_CONTRACT_VIOLATION"


@pytest.mark.parametrize("state,with_image", [(S.CREATED, True), (S.IMAGE_RECEIVED, False), (S.COMPLETED, True)])
def test_pipeline_requires_image_received(state, with_image):
    with pytest.raises(ValueError):
        PipelineOrchestrator(STUBS).run(make_ctx(state, with_image))


def test_every_stage_records_duration():
    ctx = PipelineOrchestrator(STUBS).run(make_ctx())
    module_events = [h for h in ctx.history if h.event_type == "MODULE_RESULT"]
    assert len(module_events) == 6
    assert all(h.duration_ms is not None and h.duration_ms >= 0 for h in module_events)


def test_stubs_report_not_implemented_in_health_description():
    assert all(not m["implemented"] for m in STUBS.describe().values())
    assert set(STUBS.describe()) == {"quality_gate", "cv_engine", "decision_engine", "pipeline_analyzer",
                                     "rca_engine", "action_engine"}
