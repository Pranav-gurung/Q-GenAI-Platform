"""Canonical end-to-end regression (no live LLM API): the whole chain stays connected.

RECEIVING / STORAGE / PICKING PASS, PACKING leak:
CV -> rules decision -> fulfillment analysis -> GenAI RCA (mock provider) -> RAG / SOP -> corrective actions
-> operator feedback -> learning analysis -> operations analytics.
"""
from tests.feedback.conftest import ai_client, canonical, demo_index  # noqa: F401


def test_canonical_chain(ai_client):  # noqa: F811
    c = ai_client
    k = canonical(c)
    iid, r = k["inspection_id"], k["result"]

    assert {d["defect_type"] for d in r["cv"]["detections"]} == {"LEAKING_PACKAGE"}                    # CV
    assert k["decision"] == "FAIL" and r["decision"]["outcome"] == "FAIL"                              # decision
    (a,) = r["pipeline_analysis"]["trace_analysis"]["defect_analyses"]                                 # pipeline
    assert (a["first_observed_stage"], a["last_known_good_stage"], a["evidence_level"], a["classification"]) == (
        "PACKING", "PICKING", "HIGH", "NEW_DEFECT")
    rca = r["rca"]["root_cause_analysis"]                                                              # GenAI
    assert rca["status"] == "COMPLETED" and rca["defects"][0]["likely_causes"]
    assert all(any(w in c_["hypothesis"].lower() for w in ("possible", "likely", "may", "could"))
               for c_ in rca["defects"][0]["likely_causes"])
    assert "does not prove physical causality" in rca["disclaimer"]
    rag = rca["rag"]                                                                                   # RAG
    assert rag["status"] == "GROUNDED"
    assert any((g["source_document"], g["source_section"], g["source_version"]) ==
               ("Packing and Sealing SOP", "Seal Inspection", "demo-1.0") for g in rag["grounded_actions"])
    assert any(a_["sop_reference"] == "PACKING-SOP demo-1.0 §Seal Inspection" for a_ in r["action"]["actions"])
    assert k["status"] == "COMPLETED"

    fb = c.post(f"/api/inspections/{iid}/feedback", json={                                            # operator
        "operator_id": "OP-17", "decision_correct": True, "defect_correct": True, "stage_correct": True,
        "rca_useful": True, "action_useful": True, "comment": "Recommendation matches the packing SOP."})
    assert fb.status_code == 201 and fb.json()["corrections"] == []
    assert c.get(f"/api/inspections/{iid}/feedback").json()[0]["feedback_id"] == fb.json()["feedback_id"]
    assert c.get("/api/feedback/analysis").json()["rca"]["useful_rate"] == 1.0                        # learning

    o = c.get("/api/analytics/overview").json()                                                        # analytics
    assert (o["inspections"]["total"], o["inspections"]["fail"], o["inspections"]["pass"]) == (4, 1, 3)
    assert o["hotspots"][0]["stage"] == "PACKING" and o["defects"][0]["defect_type"] == "LEAKING_PACKAGE"
    assert o["investigation"]["stage"] == "PACKING"
    assert o["feedback"]["sop_grounded_actions"] == {"rated": 1, "useful": 1, "not_useful": 0, "useful_rate": 1.0}

    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["rag"]["status"] == "ready"
    assert h["feedback"]["status"] == h["analytics"]["status"] == "ready"
