"""End-to-end with REAL CV inference: RECEIVING/STORAGE/PICKING pass, leakage at PACKING ->
CV -> rules -> fulfillment analysis -> GenAI RCA (mock LLM, records exactly what it receives) -> actions."""
import io

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("ultralytics")

from backend.db.base import Base  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.rca.guardrails import CERTAINTY  # noqa: E402
from backend.rca.prompt import evidence_from_user_message  # noqa: E402
from backend.services.reference_service import seed_sample_data  # noqa: E402
from tests.cv.conftest import DEMO_ARTIFACT, DEMO_DATASET  # noqa: E402
from tests.cv.helpers import textured  # noqa: E402

GENAI = {"pipeline_analysis_engine": "deterministic", "rca_engine": "genai", "action_engine": "rca",
         "llm_enabled": True, "llm_provider": "mock", "decision_auto_pass": True}


def client_for(settings):
    app = create_app(settings.model_copy(update=GENAI))
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as s:
        seed_sample_data(s, settings.sample_data_path)
    return TestClient(app)


def upload(c, stage, item, data, name="p.jpg", ctype="image/jpeg"):
    iid = c.post("/api/inspections", json={"store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML", "item_id": item,
                                           "stage": stage}).json()["inspection_id"]
    r = c.post(f"/api/inspections/{iid}/image", files={"file": (name, data, ctype)})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.skipif(not DEMO_ARTIFACT.is_file() or not DEMO_DATASET.is_file(), reason="demo model not trained")
def test_canonical_demo_end_to_end(cv_settings):
    root = DEMO_DATASET.parent
    labels = {p.stem: p.read_text().split() for p in sorted((root / "labels/test").glob("*.txt"))}
    clean = [s for s, v in labels.items() if not v][:3]
    leak = next(s for s, v in labels.items() if v and set(v[0::5]) == {"2"})
    s = cv_settings.model_copy(update={"cv_model_path": DEMO_ARTIFACT, "cv_model_version": None})
    with client_for(s) as c:
        for stem, stage in zip(clean, ["RECEIVING", "STORAGE", "PICKING"]):
            assert upload(c, stage, "UNIT-DEMO", (root / "images/test" / f"{stem}.jpg").read_bytes())["decision"] == "PASS"
        k = upload(c, "PACKING", "UNIT-DEMO", (root / "images/test" / f"{leak}.jpg").read_bytes())
        sent = evidence_from_user_message(c.app.state.rca_service.provider.requests[-1].user)
        again = c.get(f"/api/inspections/{k['inspection_id']}/rca").json()

    # CV + rules (deterministic) decided the FAIL
    det = k["result"]["cv"]["detections"]
    assert k["decision"] == "FAIL" and {d["model_class"] for d in det} == {"leakage"} and det[0]["severity"] == "HIGH"
    # Part 3 (deterministic)
    pa = k["result"]["pipeline_analysis"]["trace_analysis"]["defect_analyses"][0]
    assert (pa["first_observed_stage"], pa["last_known_good_stage"], pa["evidence_level"]) == ("PACKING", "PICKING", "HIGH")
    # the RCA layer received exactly that evidence
    a = sent["defect_analyses"][0]
    assert (a["first_observed_stage"], a["last_known_good_stage"], a["evidence_level"], a["classification"]) == (
        "PACKING", "PICKING", "HIGH", "NEW_DEFECT")
    assert [t["status"] for t in sent["timeline"]] == ["PASS", "PASS", "PASS", "FAIL", "NOT_OBSERVED", "NOT_OBSERVED"]
    assert sent["current_defects"][0]["cv_confidence"] == det[0]["confidence"]
    # RCA output: facts vs hypotheses, no causality claim, actions present
    rca = k["result"]["rca"]["root_cause_analysis"]
    d = rca["defects"][0]
    assert rca["status"] == "COMPLETED" and rca["requires_review"] is False
    assert d["facts"]["first_observed_stage"] == "PACKING" and d["facts"]["last_known_good_stage"] == "PICKING"
    assert "The last inspection that explicitly passed was PICKING." in d["observed_evidence"]
    assert d["likely_causes"] and all(c["hypothesis"].startswith("Possible") for c in d["likely_causes"])
    assert not any(CERTAINTY.search(c["hypothesis"] + c["rationale"]) for c in d["likely_causes"])
    assert "does not prove physical causality" in d["limitations"][0]
    assert k["result"]["action"]["status"] == "COMPLETED" and k["result"]["action"]["actions"]
    assert k["status"] == "COMPLETED"  # FAIL + corrective actions + HIGH evidence
    assert again["status"] == "COMPLETED" and again["defects"][0]["facts"] == d["facts"]  # GET /rca recomputes the same facts


def test_real_inference_then_operator_fail_then_rca(cv_settings):
    """Always runs (untrained fixture model): real inference -> PASS observations, operator corrects PACKING."""
    def png(seed):
        buf = io.BytesIO()
        textured((320, 240), seed=seed).save(buf, format="PNG")
        return buf.getvalue()
    with client_for(cv_settings) as c:
        for i, st in enumerate(["RECEIVING", "STORAGE", "PICKING"]):
            assert upload(c, st, "UNIT-FX", png(i), "p.png", "image/png")["decision"] == "PASS"
        k = upload(c, "PACKING", "UNIT-FX", png(7), "p.png", "image/png")
        c.post(f"/api/inspections/{k['inspection_id']}/feedback",
               json={"operator_id": "OP-1", "outcome": "FAIL", "defect_types": ["LEAKING_PACKAGE"]})
        rca = c.get(f"/api/inspections/{k['inspection_id']}/rca").json()
    d = rca["defects"][0]
    assert rca["status"] == "COMPLETED" and d["facts"]["evidence_level"] == "HIGH"
    assert d["facts"]["cv_confidence"] is None and "recorded by an operator" in " ".join(d["observed_evidence"])
