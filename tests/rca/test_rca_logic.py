"""RCA behaviour per Part 3 scenario (mock provider: deterministic, offline)."""
from backend.core.config import Settings
from backend.domain.fulfillment import FulfillmentStage as St
from backend.rca.service import RCAService
from backend.schemas.fulfillment import EvidenceSource
from backend.schemas.rca import CAUSALITY_DISCLAIMER, RCAStatus
from tests.fulfillment.helpers import CRUSHED, LABEL, LEAK, D, K, P, R, S, obs
from tests.rca.helpers import CANONICAL, service, trace_of


def test_high_evidence_new_defect_canonical_demo():
    svc, mock = service()
    rca = svc.analyze(trace_of(CANONICAL))
    assert rca.status is RCAStatus.COMPLETED and not rca.requires_review and rca.generation.ai_generated
    (d,) = rca.defects
    f = d.facts
    assert (f.first_observed_stage, f.last_known_good_stage, f.evidence_level.value, f.classification.value) == (
        K, P, "HIGH", "NEW_DEFECT")
    assert "LEAKING_PACKAGE was first observed at PACKING." in d.observed_evidence
    assert "The last inspection that explicitly passed was PICKING." in d.observed_evidence
    assert {c.category.value for c in d.likely_causes} >= {"SEALING_PROCESS", "PACKAGING_MATERIAL", "HANDLING"}
    assert all(c.related_stage in (None, K) for c in d.likely_causes)
    assert d.corrective_actions and d.verification_steps
    assert d.limitations[0] == CAUSALITY_DISCLAIMER and "does not prove physical causality" in rca.disclaimer
    assert len(mock.requests) == 1


def test_pre_existing_defect_is_not_blamed_on_later_stages():
    svc, _ = service()
    rca = svc.analyze(trace_of([obs(R, "FAIL", [LEAK]), obs(S, "FAIL", [LEAK]), obs(K, "FAIL", [LEAK])]))
    (d,) = rca.defects
    assert d.facts.classification.value == "PRE_EXISTING" and d.facts.current_stage_attribution.value == "NOT_SUPPORTED"
    assert all(c.related_stage is None for c in d.likely_causes)
    assert not any("PACKING" in c.hypothesis.upper() for c in d.likely_causes)
    assert any("does not support attributing it to any later stage" in x for x in d.limitations)
    assert any("persistence of the same defect type" in x for x in d.observed_evidence)


def test_missing_intermediate_stages_acknowledged():
    svc, _ = service()
    rca = svc.analyze(trace_of([obs(R), obs(K, "FAIL", [LEAK])]))
    (d,) = rca.defects
    assert d.facts.missing_stages == [S, P] and rca.requires_review
    assert any("STORAGE, PICKING were not inspected, so the available evidence does not establish when the defect "
               "physically occurred" in x for x in d.limitations)
    assert "Not inspected before the first observation: STORAGE, PICKING." in d.observed_evidence
    assert d.verification_steps[0].startswith("Review the missing or uncertain inspection evidence")


def test_review_previous_stage():
    svc, _ = service()
    rca = svc.analyze(trace_of([obs(R), obs(S), obs(P, "REVIEW"), obs(K, "FAIL", [LEAK])]))
    (d,) = rca.defects
    assert d.facts.classification.value == "UNCERTAIN" and rca.requires_review
    assert any("required review; it is not treated as a pass" in x for x in d.observed_evidence)
    assert "PICKING: REVIEW" in d.observed_evidence[0]


def test_multiple_defects_analysed_separately():
    svc, mock = service()
    rca = svc.analyze(trace_of([obs(R), obs(S), obs(P), obs(K, "FAIL", [CRUSHED]), obs(D, "FAIL", [CRUSHED, LEAK])], current=D))
    assert [d.defect_type for d in rca.defects] == [CRUSHED, LEAK]
    assert {d.facts.first_observed_stage for d in rca.defects} == {K, D}
    sent = mock.requests[0].output_schema["properties"]["findings"]["items"]["properties"]["defect_type"]["enum"]
    assert sent == ["CRUSHED_PACKAGE", "LEAKING_PACKAGE"]


def test_insufficient_observations():
    svc, _ = service()
    rca = svc.analyze(trace_of([obs(K, "FAIL", [LABEL])]))
    (d,) = rca.defects
    assert d.facts.explanation_code.value == "NO_PRIOR_OBSERVATION" and rca.requires_review
    assert any("no earlier inspection" in x for x in d.limitations)


def test_no_defect_is_not_applicable_without_llm_call():
    svc, mock = service()
    rca = svc.analyze(trace_of([obs(R), obs(S), obs(P), obs(K)]))
    assert rca.status is RCAStatus.NOT_APPLICABLE and rca.defects == [] and mock.requests == []


def test_operator_feedback_overrides_decision():
    svc, mock = service()
    rca = svc.analyze(trace_of([obs(R), obs(S), obs(P), obs(K, "FAIL", [CRUSHED], source=EvidenceSource.OPERATOR)]))
    (d,) = rca.defects
    assert d.facts.cv_confidence is None and "The defect at PACKING was recorded by an operator." in d.observed_evidence
    from backend.rca.prompt import evidence_from_user_message
    ev = evidence_from_user_message(mock.requests[0].user)
    assert ev["current_decision_source"] == "OPERATOR" and ev["current_defects"][0]["source"] == "OPERATOR"


def test_disabled_llm_keeps_deterministic_facts():
    rca = RCAService(Settings(_env_file=None)).analyze(trace_of(CANONICAL))
    assert rca.status is RCAStatus.UNAVAILABLE and rca.generation.error_code == "LLM_DISABLED" and rca.requires_review
    assert rca.summary is None and rca.defects[0].likely_causes == []
    assert rca.defects[0].facts.first_observed_stage is St.PACKING and rca.defects[0].observed_evidence


def test_provider_failures_degrade_safely():
    for mode, code in [("timeout", "LLM_TIMEOUT"), ("unavailable", "LLM_UNAVAILABLE"), ("rate_limited", "LLM_RATE_LIMITED"),
                       ("api_error", "LLM_API_ERROR"), ("refused", "LLM_REFUSED")]:
        rca = service(mode)[0].analyze(trace_of(CANONICAL))
        assert rca.status is RCAStatus.UNAVAILABLE and rca.generation.error_code == code and rca.requires_review


def test_unexpected_provider_exception_is_contained():
    svc, mock = service()
    mock.generate_rca = lambda prompt: (_ for _ in ()).throw(RuntimeError("bug with secret sk-ant-xyz"))
    rca = svc.analyze(trace_of(CANONICAL))
    assert rca.status is RCAStatus.UNAVAILABLE and rca.generation.error_code == "LLM_PROVIDER_ERROR"
    assert "sk-ant" not in rca.model_dump_json()
