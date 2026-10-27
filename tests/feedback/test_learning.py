"""Error analysis, learning-candidate export, and the guarantee that feedback changes no production behaviour."""
import json

from backend.core.config import get_settings
from scripts import export_learning_candidates
from tests.feedback.conftest import LEAK_COLOR, canonical, inspect


def review(c, iid, **body):
    r = c.post(f"/api/inspections/{iid}/feedback", json={"operator_id": "OP-1", **body})
    assert r.status_code == 201, r.text
    return r.json()


def seed_reviews(c):
    """Four reviewed items + one PASS inspection with a plain verdict."""
    a = canonical(c, "UNIT-A")["inspection_id"]  # all correct
    review(c, a, decision_correct=True, defect_correct=True, stage_correct=True, rca_useful=True, action_useful=True,
           comment="Recommendation matches the packing SOP.")
    b = canonical(c, "UNIT-B")["inspection_id"]  # stage corrected
    review(c, b, decision_correct=True, defect_correct=True, stage_correct=False, suspected_stage="PICKING",
           rca_useful=True, action_useful=False)
    d = canonical(c, "UNIT-C")["inspection_id"]  # defect + decision corrected
    review(c, d, decision_correct=False, outcome="REVIEW", defect_correct=False, defect_types=["TORN_PACKAGE"],
           rca_useful=False)
    e = canonical(c, "UNIT-D")["inspection_id"]  # first reviewed as correct, then superseded by a stage correction
    review(c, e, decision_correct=True, stage_correct=True)
    review(c, e, decision_correct=True, stage_correct=False, suspected_stage="STORAGE", action_useful=True)
    return a, b, d, e


def test_error_analysis_counts(ai_client):
    seed_reviews(ai_client)
    s = ai_client.get("/api/feedback/analysis").json()
    assert s["feedback_submissions"] == 5 and s["inspections_reviewed"] == 4 and s["scope"].startswith("latest")
    assert (s["decisions"]["reviewed"], s["decisions"]["corrected"]) == (4, 1)
    assert s["decisions"]["transitions"] == {"FAIL -> REVIEW": 1} and s["decisions"]["correction_rate"] == 0.25
    assert (s["defects"]["reviewed"], s["defects"]["corrected"]) == (3, 1)
    assert s["defects"]["by_ai_value"]["LEAKING_PACKAGE"]["corrected"] == 1
    assert s["defects"]["transitions"] == {"LEAKING_PACKAGE -> TORN_PACKAGE": 1}
    assert (s["stages"]["reviewed"], s["stages"]["corrected"]) == (3, 2)  # superseded "correct" is not counted
    assert s["stages"]["by_ai_value"]["PACKING"] == {"reviewed": 3, "confirmed": 1, "corrected": 2,
                                                     "correction_rate": 0.6667}
    assert s["stages"]["transitions"] == {"PACKING -> PICKING": 1, "PACKING -> STORAGE": 1}
    assert s["rca"] == {"rated": 3, "useful": 2, "not_useful": 1, "useful_rate": 0.6667}
    assert s["actions"] == {"rated": 3, "useful": 2, "not_useful": 1, "useful_rate": 0.6667}
    assert s["learning_status"] == {"PENDING": 4}
    loop = s["learning_loop"]
    assert loop["automatic_retraining"] is False and loop["automatic_rule_changes"] is False
    assert loop["flow"][-2:] == ["Human / ML engineer review", "Future model version"]


def test_analysis_filters_and_empty_state(ai_client):
    empty = ai_client.get("/api/feedback/analysis").json()
    assert empty["inspections_reviewed"] == 0 and empty["decisions"]["correction_rate"] is None
    seed_reviews(ai_client)
    assert ai_client.get("/api/feedback/analysis?store_id=DS-MUM-001").json()["inspections_reviewed"] == 0
    assert ai_client.get("/api/feedback/analysis?store_id=bad id").status_code == 422


def test_learning_candidates_keep_ai_and_operator_values(ai_client):
    a, b, d, e = seed_reviews(ai_client)
    rows = {r["inspection_id"]: r for r in ai_client.get("/api/feedback/learning-candidates").json()}
    assert set(rows) == {a, b, d, e}
    r = rows[b]
    assert (r["ai_stage"], r["operator_stage"], r["correction_layers"]) == ("PACKING", "PICKING", ["STAGE"])
    assert (r["ai_decision"], r["operator_decision"], r["ai_defects"]) == ("FAIL", "FAIL", ["LEAKING_PACKAGE"])
    assert r["item_id"] == "UNIT-B" and r["image_sha256"] and r["inspection_stage"] == "PACKING"
    assert r["source"] == "operator_feedback" and r["requires_curation"] is True and r["learning_status"] == "PENDING"
    assert rows[d]["correction_layers"] == ["DECISION", "DEFECT"]
    assert (rows[d]["ai_defects"], rows[d]["operator_defects"]) == (["LEAKING_PACKAGE"], ["TORN_PACKAGE"])
    assert rows[a]["correction_layers"] == [] and rows[a]["comment"] == "Recommendation matches the packing SOP."
    only = ai_client.get("/api/feedback/learning-candidates?corrections_only=true").json()
    assert {x["inspection_id"] for x in only} == {b, d, e}


def test_unlabelled_defects_are_not_invented(ai_client):
    k = inspect(ai_client, "PACKING", LEAK_COLOR, item="UNIT-X")
    review(ai_client, k["inspection_id"], outcome="FAIL")  # verdict only, no defect label
    (row,) = ai_client.get("/api/feedback/learning-candidates").json()
    assert row["operator_defects"] is None and row["ai_defects"] == ["LEAKING_PACKAGE"]


def test_export_cli_writes_jsonl(ai_client, settings, tmp_path, monkeypatch):
    seed_reviews(ai_client)
    monkeypatch.setenv("DATABASE_URL", settings.database_url)
    get_settings.cache_clear()
    try:
        out = tmp_path / "candidates.jsonl"
        assert export_learning_candidates.main(["--out", str(out), "--corrections-only"]) == 0
    finally:
        get_settings.cache_clear()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3 and all(r["requires_curation"] for r in rows)


def test_feedback_never_changes_production_behaviour(ai_client):
    """Many operators say FAIL should be REVIEW: the next identical inspection is still decided by the rules."""
    for i in range(5):
        k = inspect(ai_client, "PACKING", (220 + i, 40, 40), item=f"UNIT-OVR-{i}")
        review(ai_client, k["inspection_id"], decision_correct=False, outcome="REVIEW", defect_correct=False,
               defect_types=["TORN_PACKAGE"])
    fresh = inspect(ai_client, "PACKING", (230, 40, 40), item="UNIT-NEW")
    assert fresh["decision"] == "FAIL"
    assert fresh["result"]["cv"]["detections"][0]["defect_type"] == "LEAKING_PACKAGE"
    assert fresh["result"]["cv"]["module_version"] == "test-cv-1"  # same model version
    assert ai_client.get("/api/feedback/analysis").json()["decisions"]["corrected"] == 5
