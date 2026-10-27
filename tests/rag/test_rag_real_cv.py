"""Canonical demo with REAL CV: trained YOLO detects the leak at PACKING -> Part 3 -> RCA -> SOP-grounded actions.
SYNTHETIC / DEMO images and documents; skipped when the demo model has not been trained."""
import pytest

pytest.importorskip("ultralytics")

from tests.cv.conftest import DEMO_ARTIFACT, DEMO_DATASET, cv_settings, fixture_model  # noqa: E402,F401
from tests.cv.test_fulfillment_full_path import client_for, upload  # noqa: E402
from tests.rag.helpers import KNOWLEDGE  # noqa: E402


@pytest.mark.skipif(not DEMO_ARTIFACT.is_file() or not DEMO_DATASET.is_file(), reason="demo model not trained")
def test_canonical_leak_with_real_cv_is_sop_grounded(cv_settings, demo_index):  # noqa: F811
    root = DEMO_DATASET.parent
    labels = {p.stem: p.read_text().split() for p in sorted((root / "labels/test").glob("*.txt"))}
    clean = [s for s, v in labels.items() if not v][:3]
    leak = next(s for s, v in labels.items() if v and set(v[0::5]) == {"2"})  # leakage-only image
    s = cv_settings.model_copy(update={
        "cv_model_path": DEMO_ARTIFACT, "cv_model_version": None, "rca_engine": "genai", "action_engine": "rca",
        "llm_enabled": True, "llm_provider": "mock", "rag_enabled": True, "rag_index_path": demo_index,
        "rag_knowledge_path": KNOWLEDGE})
    with client_for(s) as c:
        for stem, stage in zip(clean, ["RECEIVING", "STORAGE", "PICKING"]):
            assert upload(c, stage, (root / "images/test" / f"{stem}.jpg").read_bytes(), "c.jpg",
                          "image/jpeg")["decision"] == "PASS"
        k = upload(c, "PACKING", (root / "images/test" / f"{leak}.jpg").read_bytes(), "l.jpg", "image/jpeg")
    assert k["decision"] == "FAIL" and {x["model_class"] for x in k["result"]["cv"]["detections"]} == {"leakage"}
    rca = k["result"]["rca"]["root_cause_analysis"]
    rag = rca["rag"]
    assert rca["defects"][0]["facts"]["first_observed_stage"] == "PACKING"
    assert rag["status"] == "GROUNDED"
    assert "PACKING-SOP#seal-inspection" in [x["chunk_id"] for x in rag["retrieved_sources"]]
    assert any((g["source_document"], g["source_section"], g["source_version"]) ==
               ("Packing and Sealing SOP", "Seal Inspection", "demo-1.0") for g in rag["grounded_actions"])
    actions = k["result"]["action"]["actions"]
    assert any(a["sop_reference"] == "PACKING-SOP demo-1.0 §Seal Inspection" for a in actions)
    assert k["status"] == "COMPLETED"
