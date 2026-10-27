"""End-to-end: API upload -> storage -> quality gate -> REAL YOLO inference -> CVResult
-> rules decision -> pipeline analysis -> persistence -> GET / health. Nothing in the CV path is mocked
except where a test explicitly simulates a failure."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

pytest.importorskip("ultralytics")

from backend.db.base import Base  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.services.reference_service import seed_sample_data  # noqa: E402
from tests.cv.conftest import DEMO_ARTIFACT, DEMO_DATASET, FIXTURE_VERSION  # noqa: E402
from tests.cv.helpers import textured  # noqa: E402

PAYLOAD = {"store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML", "stage": "PACKING"}


def png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def client_for(settings):
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as s:
        seed_sample_data(s, settings.sample_data_path)
    return TestClient(app)


def run(client, data: bytes, name="p.png", ctype="image/png"):
    ins = client.post("/api/inspections", json=PAYLOAD).json()
    r = client.post(f"/api/inspections/{ins['inspection_id']}/image", files={"file": (name, data, ctype)})
    assert r.status_code == 200, r.text
    return r.json()


def stages(body):
    return [h["to_state"] for h in body["history"] if h["event_type"] == "STATE_TRANSITION"]


def test_health_reports_real_cv_engine(cv_settings):
    with client_for(cv_settings) as c:
        h = c.get("/api/health").json()
    cv = h["pipeline_modules"]["cv_engine"]
    assert h["status"] == "ok"
    assert cv["name"] == "yolov8" and cv["implemented"] and cv["model_loaded"] and cv["model_version"] == FIXTURE_VERSION
    assert cv["device"] == "cpu"
    assert h["pipeline_modules"]["quality_gate"]["name"] == "basic"
    assert h["pipeline_modules"]["decision_engine"]["name"] == "rules"


@pytest.mark.parametrize("auto_pass,status,decision", [(True, "COMPLETED", "PASS"),
                                                        (False, "REQUIRES_REVIEW", "REVIEW")])
def test_full_path_real_inference_no_detections(cv_settings, auto_pass, status, decision):
    with client_for(cv_settings.model_copy(update={"decision_auto_pass": auto_pass})) as c:
        body = run(c, png(textured((320, 240))))
        again = c.get(f"/api/inspections/{body['inspection_id']}").json()
    res = body["result"]
    assert res["quality"]["status"] == "COMPLETED" and res["quality"]["acceptable"] is True
    assert res["cv"]["status"] == "COMPLETED" and res["cv"]["inference"]["inference_device"] == "cpu"
    assert res["cv"]["detections"] == []
    assert res["decision"]["status"] == "COMPLETED" and res["decision"]["outcome"] == decision
    assert body["status"] == status and body["decision"] == decision
    assert body["model_versions"]["cv"] == f"yolov8:{FIXTURE_VERSION}"
    assert stages(body)[:5] == ["CREATED", "IMAGE_RECEIVED", "QUALITY_CHECK", "CV_PROCESSING", "DECISION"]
    assert again["result"]["cv"]["inference"]["model_version"] == FIXTURE_VERSION  # persisted


def test_full_path_with_detections_goes_through_pipeline_analysis(cv_settings):
    with client_for(cv_settings.model_copy(update={"cv_confidence_threshold": 0.0})) as c:
        body = run(c, png(textured((320, 240))))
        app_db = c.app.state.session_factory
        from backend.db.models import InspectionResult
        with app_db() as s:
            row = s.get(InspectionResult, body["result_id"])
            assert row.cv_model_version == FIXTURE_VERSION
    dets = body["result"]["cv"]["detections"]
    assert dets and all(d["confidence"] < 0.5 for d in dets)
    # confident-FAIL criteria not met by an untrained model -> REVIEW decision, never FAIL/PASS
    assert body["result"]["decision"]["outcome"] == "REVIEW"
    assert body["status"] == "REQUIRES_REVIEW"


def test_corrupt_upload_still_rejected_cleanly_with_cv_loaded(cv_settings):
    """Regression: Ultralytics' global PIL patch must not turn CORRUPT_IMAGE (422) into a 500."""
    with client_for(cv_settings) as c:
        ins = c.post("/api/inspections", json=PAYLOAD).json()
        r = c.post(f"/api/inspections/{ins['inspection_id']}/image", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "CORRUPT_IMAGE"


def test_quality_gate_rejects_before_cv(cv_settings):
    blurry = textured((320, 240)).filter(ImageFilter.GaussianBlur(8))
    with client_for(cv_settings) as c:
        body = run(c, png(blurry))
    assert "BLURRY" in body["result"]["quality"]["issues"]
    assert body["result"]["cv"] is None and body["status"] == "REQUIRES_REVIEW"
    assert body["history"][-1]["error_code"] == "IMAGE_QUALITY_REJECTED"


def test_missing_model_never_passes(cv_settings, tmp_path):
    s = cv_settings.model_copy(update={"cv_model_path": tmp_path / "missing.pt", "decision_auto_pass": True})
    with client_for(s) as c:
        health = c.get("/api/health").json()
        body = run(c, png(textured((320, 240))))
    assert health["status"] == "degraded" and health["pipeline_modules"]["cv_engine"]["available"] is False
    assert body["result"]["cv"]["status"] == "UNAVAILABLE"
    assert body["result"]["decision"]["status"] == "SKIPPED"
    assert body["decision"] is None and body["status"] == "REQUIRES_REVIEW"
    assert any(h["error_code"] == "MODULE_UNAVAILABLE" for h in body["history"])
    assert body["model_versions"] == {"quality": "basic:1"}


def test_inference_crash_fails_inspection_without_leaking(cv_settings, monkeypatch):
    with client_for(cv_settings) as c:
        engine = c.app.state.orchestrator.modules.cv_engine

        def boom(bgr):
            raise RuntimeError("CUDA OOM at 0xdeadbeef")
        monkeypatch.setattr(engine, "_predict", boom)
        body = run(c, png(textured((320, 240))))
    assert body["status"] == "FAILED" and body["error"]["code"] == "MODULE_ERROR"
    assert "deadbeef" not in str(body)


@pytest.mark.skipif(not DEMO_ARTIFACT.is_file() or not DEMO_DATASET.is_file(),
                    reason="demo model not trained (python -m ml.cv.synthetic && python -m ml.cv.train ...)")
def test_trained_demo_model_detects_synthetic_defects(cv_settings):
    """SYNTHETIC / DEMO check: the trained artifact finds defects in held-out synthetic test images."""
    from pathlib import Path
    from ml.cv.taxonomy import load_taxonomy
    names = load_taxonomy().names
    s = cv_settings.model_copy(update={"cv_model_path": DEMO_ARTIFACT, "cv_model_version": None})
    test_imgs = sorted((DEMO_DATASET.parent / "images/test").glob("*.jpg"))
    with client_for(s) as c:
        hits = total = 0
        for img in test_imgs:
            label = (DEMO_DATASET.parent / "labels/test" / f"{img.stem}.txt").read_text().split()
            if not label:
                continue
            gt = {names[int(label[i])] for i in range(0, len(label), 5)}
            body = run(c, img.read_bytes(), name="t.jpg", ctype="image/jpeg")
            found = {d["model_class"] for d in body["result"]["cv"]["detections"]}
            total += 1
            hits += bool(gt & found)
            if total >= 25:
                break
        assert body["result"]["cv"]["inference"]["model_version"] == "qgenai-yolov8n-synthetic-v1"
    assert total >= 20 and hits / total >= 0.8, (hits, total)
