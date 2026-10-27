"""Canonical demo with REAL CV, then operator review: YOLO leak at PACKING -> FAIL -> PACKING / HIGH -> RCA -> SOP ->
operator confirms everything (item A) or corrects the stage to PICKING (item B) -> error analysis.
SYNTHETIC / DEMO images and SOPs; skipped when the demo model has not been trained."""
import pytest

pytest.importorskip("ultralytics")

from tests.cv.conftest import DEMO_ARTIFACT, DEMO_DATASET, cv_settings, fixture_model  # noqa: E402,F401
from tests.cv.test_fulfillment_full_path import client_for, upload  # noqa: E402
from tests.rag.conftest import demo_index  # noqa: E402,F401
from tests.rag.helpers import KNOWLEDGE  # noqa: E402


@pytest.mark.skipif(not DEMO_ARTIFACT.is_file() or not DEMO_DATASET.is_file(), reason="demo model not trained")
def test_canonical_real_cv_then_operator_review(cv_settings, demo_index):  # noqa: F811
    root = DEMO_DATASET.parent
    labels = {p.stem: p.read_text().split() for p in sorted((root / "labels/test").glob("*.txt"))}
    clean = [s for s, v in labels.items() if not v][:6]
    leaks = [s for s, v in labels.items() if v and set(v[0::5]) == {"2"}][:2]  # leakage-only images
    s = cv_settings.model_copy(update={
        "cv_model_path": DEMO_ARTIFACT, "cv_model_version": None, "rca_engine": "genai", "action_engine": "rca",
        "llm_enabled": True, "llm_provider": "mock", "rag_enabled": True, "rag_index_path": demo_index,
        "rag_knowledge_path": KNOWLEDGE})
    img = lambda stem: (root / "images/test" / f"{stem}.jpg").read_bytes()  # noqa: E731
    with client_for(s) as c:
        packing = []
        for item, cleans, leak in (("UNIT-A", clean[:3], leaks[0]), ("UNIT-B", clean[3:], leaks[1])):
            for stem, stage in zip(cleans, ["RECEIVING", "STORAGE", "PICKING"]):
                iid = c.post("/api/inspections", json={"store_id": "DS-BLR-001", "sku_id": "SKU-EGGS-6",
                                                       "item_id": item, "stage": stage}).json()["inspection_id"]
                r = c.post(f"/api/inspections/{iid}/image", files={"file": ("c.jpg", img(stem), "image/jpeg")}).json()
                assert r["decision"] == "PASS"
            iid = c.post("/api/inspections", json={"store_id": "DS-BLR-001", "sku_id": "SKU-EGGS-6", "item_id": item,
                                                   "stage": "PACKING"}).json()["inspection_id"]
            k = c.post(f"/api/inspections/{iid}/image", files={"file": ("l.jpg", img(leak), "image/jpeg")}).json()
            assert k["decision"] == "FAIL"
            assert {d["defect_type"] for d in k["result"]["cv"]["detections"]} == {"LEAKING_PACKAGE"}
            assert k["result"]["rca"]["root_cause_analysis"]["rag"]["status"] == "GROUNDED"
            packing.append(iid)

        base = {"operator_id": "OP-17", "decision_correct": True, "defect_correct": True, "rca_useful": True,
                "action_useful": True}
        a = c.post(f"/api/inspections/{packing[0]}/feedback", json={
            **base, "stage_correct": True, "comment": "Recommendation matches the packing SOP."}).json()
        b = c.post(f"/api/inspections/{packing[1]}/feedback", json={
            **base, "stage_correct": False, "suspected_stage": "PICKING",
            "comment": "Leak was visible during picking inspection."}).json()
        analysis = c.get("/api/feedback/analysis").json()
        candidates = c.get("/api/feedback/learning-candidates?corrections_only=true").json()

    assert a["corrections"] == [] and a["ai_snapshot"]["pipeline"]["defects"][0]["evidence_level"] == "HIGH"
    assert a["ai_snapshot"]["defects"] == ["LEAKING_PACKAGE"] and a["model_versions"]["cv"]
    assert b["corrections"] == [{"layer": "STAGE", "ai_value": "PACKING", "operator_value": "PICKING"}]
    assert analysis["stages"]["transitions"] == {"PACKING -> PICKING": 1} and analysis["decisions"]["corrected"] == 0
    assert analysis["rca"]["useful_rate"] == 1.0
    (cand,) = candidates
    assert (cand["item_id"], cand["ai_stage"], cand["operator_stage"]) == ("UNIT-B", "PACKING", "PICKING")
