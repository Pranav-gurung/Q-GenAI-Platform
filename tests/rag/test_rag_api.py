"""RAG through the API and the orchestrator: GET /rca, /health, action engine, routing, end-to-end canonical demo."""
import json
from dataclasses import replace

import pytest

from backend.core.config import Settings
from backend.domain.enums import DecisionOutcome
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.registry import build_modules
from backend.pipeline.states import PipelineState as S
from backend.rca.engine import GenAIRCAEngine
from backend.rca.providers.mock_provider import MockProvider
from backend.rca.service import RCAService
from tests.fulfillment.helpers import P, R, S as STO, obs
from tests.rag.helpers import KNOWLEDGE
from tests.rca.test_rca_api import SECRET, canonical, make_client
from tests.rca.test_rca_pipeline_routing import LeakCV, replace_ctx
from tests.unit.test_orchestrator import FixedDecision, make_ctx

SEAL = ("Packing and Sealing SOP", "Seal Inspection", "demo-1.0")


@pytest.fixture
def rag_client(settings, demo_index):
    with make_client(settings, llm_enabled=True, llm_provider="mock", llm_api_key=SECRET, rag_enabled=True,
                     rag_index_path=demo_index, rag_knowledge_path=KNOWLEDGE) as c:
        yield c


def test_end_to_end_canonical_with_operator_feedback(rag_client):
    """RECEIVING/STORAGE/PICKING PASS, PACKING FAIL (leak): RCA -> retrieval -> grounding -> cited actions."""
    k = canonical(rag_client)
    r = rag_client.get(f"/api/inspections/{k}/rca")
    assert r.status_code == 200, r.text
    rca = r.json()
    rag = rca["rag"]
    d = rca["defects"][0]
    assert (d["facts"]["first_observed_stage"], d["facts"]["last_known_good_stage"]) == ("PACKING", "PICKING")
    assert d["likely_causes"] and rca["status"] == "COMPLETED"  # GenAI hypotheses kept separate
    assert rag["status"] == "GROUNDED" and rag["requires_review"] is False
    assert "PACKING-SOP#seal-inspection" in [s["chunk_id"] for s in rag["retrieved_sources"]]
    assert any((g["source_document"], g["source_section"], g["source_version"]) == SEAL
               for g in rag["grounded_actions"])
    assert all(g["source_chunk_id"] and g["supporting_quote"] for g in rag["grounded_actions"])
    assert rag["unsupported_actions"] and "DEMO / SYNTHETIC" in rag["disclaimer"]
    assert "http" not in json.dumps(rag["grounded_actions"])  # no fabricated URLs
    assert SECRET not in r.text


def test_health_reports_rag_without_secrets(rag_client):
    r = rag_client.get("/api/health")
    h = r.json()
    assert h["rag"]["status"] == "ready" and h["rag"]["enabled"] is True
    assert (h["rag"]["documents"], h["rag"]["embedding_provider"], h["rag"]["vector_store"]) == (9, "hashing",
                                                                                                "local-numpy")
    rca_slot = h["pipeline_modules"]["rca_engine"]
    assert rca_slot.get("rag", {"status": "ready"})["status"] == "ready"  # present when RCA_ENGINE=genai
    assert SECRET not in r.text and "Inspect sealing equipment" not in r.text


def test_health_degraded_when_index_missing(settings, tmp_path):
    with make_client(settings, rag_enabled=True, rag_index_path=tmp_path / "none") as c:
        h = c.get("/api/health").json()
    assert h["status"] == "degraded" and h["rag"] == {**h["rag"], "status": "unavailable", "error": "INDEX_MISSING"}


def test_health_rag_disabled_by_default(client):
    assert client.get("/api/health").json()["rag"]["status"] == "disabled"


def run(prior, demo_index, rag_mode="ok"):
    s = Settings(_env_file=None, pipeline_analysis_engine="deterministic", action_engine="rca", llm_enabled=True,
                 llm_provider="mock", rag_enabled=True, rag_index_path=demo_index)
    modules = replace(build_modules(s), cv_engine=LeakCV(), decision_engine=FixedDecision(DecisionOutcome.FAIL),
                      rca_engine=GenAIRCAEngine(s, RCAService(s, provider=MockProvider(rag_mode=rag_mode))))
    return PipelineOrchestrator(modules).run(replace_ctx(make_ctx(), prior))


def test_pipeline_actions_carry_sop_references(demo_index):
    ctx = run([obs(R), obs(STO), obs(P)], demo_index)
    assert ctx.state is S.COMPLETED and ctx.action.is_real
    sop = [a for a in ctx.action.actions if a.sop_reference]
    ai = [a for a in ctx.action.actions if not a.sop_reference]
    assert sop and ai and ctx.action.actions[:len(sop)] == sop  # SOP-backed first
    assert "PACKING-SOP demo-1.0 §Seal Inspection" in {a.sop_reference for a in sop}
    assert all("not SOP-backed" in a.instructions[-1] for a in ai)
    assert all(a.instructions[-1].startswith("SOP text:") for a in sop)


def test_pipeline_requires_review_when_not_sop_grounded(demo_index):
    ctx = run([obs(R), obs(STO), obs(P)], demo_index, rag_mode="no_support")
    assert ctx.state is S.REQUIRES_REVIEW and ctx.history[-1].error_code == "SOP_GUIDANCE_REQUIRES_REVIEW"
    assert ctx.final_decision is DecisionOutcome.FAIL and all(not a.sop_reference for a in ctx.action.actions)


def test_attribution_review_still_takes_precedence(demo_index):
    ctx = run([obs(R)], demo_index)
    assert ctx.history[-1].error_code == "ATTRIBUTION_REQUIRES_REVIEW"
