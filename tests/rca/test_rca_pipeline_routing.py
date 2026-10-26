"""RCA_PROCESSING / ACTION_GENERATED in the orchestrator, with a FAIL decision (test-double decision engine)."""
from dataclasses import replace
from datetime import timedelta

from backend.core.config import Settings
from backend.domain.enums import DecisionOutcome, DefectType, ModuleStatus
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.registry import build_modules
from backend.pipeline.states import PipelineState as S
from backend.rca.engine import GenAIRCAEngine
from backend.rca.providers.mock_provider import MockProvider
from backend.rca.service import RCAService
from backend.schemas.results import CVResult, Detection
from tests.fulfillment.helpers import T0, P, R, S as STO, obs
from tests.unit.test_orchestrator import FixedDecision, make_ctx, transitions


class LeakCV:  # test double: a COMPLETED CV result with one leakage detection
    name, implemented = "test-cv", True

    def health(self):
        return {}

    def detect(self, image):
        return CVResult(module="test-cv", module_version="t", status=ModuleStatus.COMPLETED,
                        detections=[Detection(defect_type=DefectType.LEAKING_PACKAGE, confidence=0.96, severity="HIGH")])


def run(prior, mode="ok"):
    s = Settings(_env_file=None, pipeline_analysis_engine="deterministic", action_engine="rca", llm_enabled=True,
                 llm_provider="mock")
    modules = replace(build_modules(s), cv_engine=LeakCV(), decision_engine=FixedDecision(DecisionOutcome.FAIL),
                      rca_engine=GenAIRCAEngine(s, RCAService(s, provider=MockProvider(mode))))
    ctx = make_ctx()
    ctx = replace_ctx(ctx, prior)
    return PipelineOrchestrator(modules).run(ctx)


def replace_ctx(ctx, prior):
    from dataclasses import replace as dc_replace
    info = dc_replace(ctx.inspection, item_id="UNIT-1", store_id="DS-BLR-001", sku_id="SKU-MILK-500ML")
    ctx.inspection = info
    ctx.image = dc_replace(ctx.image, metadata=ctx.image.metadata.model_copy(update={"created_at": T0 + timedelta(hours=1)}))
    ctx.trace_observations = tuple(prior)
    return ctx


def test_high_evidence_fail_gets_rca_actions_and_completes():
    ctx = run([obs(R), obs(STO), obs(P)])
    assert transitions(ctx)[-3:] == [S.RCA_PROCESSING, S.ACTION_GENERATED, S.COMPLETED]
    assert ctx.rca.is_real and ctx.rca.root_cause_analysis.defects[0].facts.evidence_level.value == "HIGH"
    assert ctx.rca.possible_causes and all(c.confidence is None for c in ctx.rca.possible_causes)  # no invented numbers
    assert ctx.action.is_real and all("AI-suggested" in a.instructions[-1] for a in ctx.action.actions)
    assert ctx.final_decision is DecisionOutcome.FAIL
    assert ctx.model_versions()["rca"] == "genai:mock:mock-rca-1:rca-v1"


def test_insufficient_evidence_fail_requires_review_even_with_actions():
    ctx = run([obs(R)])  # STORAGE / PICKING never inspected
    assert ctx.action.is_real and ctx.state is S.REQUIRES_REVIEW
    assert ctx.history[-1].error_code == "ATTRIBUTION_REQUIRES_REVIEW"


def test_llm_unavailable_preserves_decision_and_analysis():
    ctx = run([obs(R), obs(STO), obs(P)], mode="timeout")
    assert ctx.rca.status is ModuleStatus.UNAVAILABLE and "LLM_TIMEOUT" in ctx.rca.message
    assert ctx.action.status is ModuleStatus.SKIPPED and ctx.state is S.REQUIRES_REVIEW
    assert ctx.history[-1].error_code == "NO_CORRECTIVE_ACTION"
    assert ctx.final_decision is DecisionOutcome.FAIL  # deterministic decision untouched
    assert ctx.pipeline_analysis.trace_analysis.defect_analyses[0].classification.value == "NEW_DEFECT"
    assert any(h.error_code == "MODULE_UNAVAILABLE" for h in ctx.history)


def test_invalid_llm_output_never_fails_inspection():
    ctx = run([obs(R), obs(STO), obs(P)], mode="unhedged")
    assert ctx.state is S.REQUIRES_REVIEW and ctx.rca.status is ModuleStatus.UNAVAILABLE
    assert "LLM_UNGROUNDED_OUTPUT" in ctx.rca.message
