"""Full path with REAL CV inference: image -> quality gate -> YOLO -> rules decision -> persisted observation
-> deterministic fulfillment analysis (pipeline stage snapshot + GET /pipeline)."""
import io

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("ultralytics")

from backend.db.base import Base  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.services.reference_service import seed_sample_data  # noqa: E402
from tests.cv.conftest import DEMO_ARTIFACT, DEMO_DATASET, FIXTURE_VERSION  # noqa: E402
from tests.cv.helpers import textured  # noqa: E402

BASE = {"store_id": "DS-BLR-001", "sku_id": "SKU-EGGS-6", "item_id": "UNIT-CV-1"}


def client_for(settings):
    app = create_app(settings.model_copy(update={"pipeline_analysis_engine": "deterministic", "decision_auto_pass": True}))
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as s:
        seed_sample_data(s, settings.sample_data_path)
    return TestClient(app)


def upload(c, stage, data, name="p.png", ctype="image/png"):
    iid = c.post("/api/inspections", json={**BASE, "stage": stage}).json()["inspection_id"]
    r = c.post(f"/api/inspections/{iid}/image", files={"file": (name, data, ctype)})
    assert r.status_code == 200, r.text
    return r.json()


def png(seed):
    buf = io.BytesIO()
    textured((320, 240), seed=seed).save(buf, format="PNG")
    return buf.getvalue()


def test_real_inference_observations_feed_stage_analysis(cv_settings):
    """Untrained fixture model: real inference finds nothing -> PASS observations (auto-pass on). The operator then
    corrects PACKING to FAIL; the trace analysis uses the persisted CV/decision results plus that correction."""
    with client_for(cv_settings) as c:
        for i, stage in enumerate(["RECEIVING", "STORAGE", "PICKING"]):
            b = upload(c, stage, png(i))
            assert b["decision"] == "PASS" and b["result"]["cv"]["status"] == "COMPLETED"
        k = upload(c, "PACKING", png(9))
        r = c.post(f"/api/inspections/{k['inspection_id']}/feedback",
                   json={"operator_id": "OP-1", "outcome": "FAIL", "defect_types": ["CRUSHED_PACKAGE"]})
        assert r.status_code == 201
        p = c.get(f"/api/inspections/{k['inspection_id']}/pipeline").json()
    obs = p["observations"]
    assert [o["decision_source"] for o in obs] == ["MODEL", "MODEL", "MODEL", "OPERATOR"]
    assert {o["model_version"] for o in obs} == {FIXTURE_VERSION}  # observations come from real persisted CV runs
    (a,) = p["defect_analyses"]
    assert (a["classification"], a["first_observed_stage"], a["last_known_good_stage"], a["evidence_level"]) == (
        "NEW_DEFECT", "PACKING", "PICKING", "HIGH")


@pytest.mark.skipif(not DEMO_ARTIFACT.is_file() or not DEMO_DATASET.is_file(), reason="demo model not trained")
def test_trained_model_detects_and_pipeline_attributes(cv_settings):
    """SYNTHETIC / DEMO images: CV itself produces the FAIL; the PIPELINE_ANALYSIS stage attributes it."""
    root = DEMO_DATASET.parent
    labels = {p.stem: p.read_text().split() for p in sorted((root / "labels/test").glob("*.txt"))}
    clean = [s for s, v in labels.items() if not v][:3]
    leaks = [s for s, v in labels.items() if v and set(v[0::5]) == {"2"}][:2]  # leakage-only images
    assert len(clean) == 3 and len(leaks) == 2
    s = cv_settings.model_copy(update={"cv_model_path": DEMO_ARTIFACT, "cv_model_version": None})
    with client_for(s) as c:
        for stem, stage in zip(clean, ["RECEIVING", "STORAGE", "PICKING"]):
            assert upload(c, stage, (root / "images/test" / f"{stem}.jpg").read_bytes(), "c.jpg", "image/jpeg")["decision"] == "PASS"
        k = upload(c, "PACKING", (root / "images/test" / f"{leaks[0]}.jpg").read_bytes(), "l.jpg", "image/jpeg")
        d = upload(c, "DISPATCH", (root / "images/test" / f"{leaks[1]}.jpg").read_bytes(), "l2.jpg", "image/jpeg")
    assert k["decision"] == "FAIL" and {x["model_class"] for x in k["result"]["cv"]["detections"]} == {"leakage"}
    snap = k["result"]["pipeline_analysis"]
    assert snap["suspected_stage"] == "PACKING"
    (a,) = snap["trace_analysis"]["defect_analyses"]
    assert (a["defect_type"], a["classification"], a["evidence_level"], a["last_known_good_stage"]) == (
        "LEAKING_PACKAGE", "NEW_DEFECT", "HIGH", "PICKING")
    (b,) = d["result"]["pipeline_analysis"]["trace_analysis"]["defect_analyses"]
    assert b["current_stage_status"] == "PERSISTING" and b["current_stage_attribution"] == "NOT_SUPPORTED"
