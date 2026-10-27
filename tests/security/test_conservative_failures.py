"""Technical failure must never turn into PASS: unavailable / failing modules end in REVIEW-type states."""
from dataclasses import replace as dc_replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.db.base import Base
from backend.main import create_app
from backend.services.reference_service import seed_sample_data
from tests.feedback.conftest import png


class ExplodingCV:
    name, implemented = "exploding-cv", True

    def health(self):
        return {}

    def detect(self, image):
        raise RuntimeError("model blew up at /secret/path/weights.pt password=hunter2")


def client_for(settings, cv=None, **update):
    app = create_app(settings.model_copy(update={"decision_engine": "rules", "decision_auto_pass": True,
                                                 "pipeline_analysis_engine": "deterministic", **update}))
    if cv is not None:
        app.state.orchestrator.modules = dc_replace(app.state.orchestrator.modules, cv_engine=cv)
    Base.metadata.create_all(app.state.engine)
    with app.state.session_factory() as db:
        seed_sample_data(db, settings.sample_data_path)
    return TestClient(app)


def run(c, data, name="a.png"):
    iid = c.post("/api/inspections", json={"store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML",
                                           "stage": "PACKING"}).json()["inspection_id"]
    return c.post(f"/api/inspections/{iid}/image", files={"file": (name, data, "image/png")})


def test_cv_exception_fails_safely_without_pass(settings):
    with client_for(settings, ExplodingCV()) as c:
        r = run(c, png((10, 200, 30)))
        a = c.get("/api/analytics/overview").json()["inspections"]
    body = r.json()
    assert r.status_code == 200 and body["status"] == "FAILED" and body["decision"] is None
    assert body["error"]["code"] == "MODULE_ERROR"
    assert "hunter2" not in r.text and "/secret/" not in r.text  # exception text is logged, never returned
    assert (a["pipeline_failed"], a["pass"], a["decided"]) == (1, 0, 0)


def test_missing_cv_model_requires_review(settings, tmp_path):
    with client_for(settings, cv_engine="yolov8", cv_model_path=tmp_path / "missing.pt") as c:
        h = c.get("/api/health").json()
        body = run(c, png((10, 200, 30))).json()
    cv = h["pipeline_modules"]["cv_engine"]
    assert h["status"] == "degraded" and cv["available"] is False and str(tmp_path) not in cv["error"]
    assert body["status"] == "REQUIRES_REVIEW" and body["decision"] is None


def test_blurred_image_is_rejected_by_the_quality_gate(settings):
    import io
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (128, 128, 128)).save(buf, "PNG")  # no edges at all
    with client_for(settings, quality_gate_engine="basic") as c:
        body = run(c, buf.getvalue()).json()
    assert body["status"] == "REQUIRES_REVIEW" and body["decision"] is None
    assert body["history"][-1]["error_code"] == "IMAGE_QUALITY_REJECTED"


def test_explicit_cuda_without_gpu_is_unavailable_not_pass(settings):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ml.cv._ultralytics import import_ultralytics
    import_ultralytics()  # undo Ultralytics' global PIL.Image.open patch (as the server does)
    if torch.cuda.is_available():
        pytest.skip("a GPU is available")
    from tests.cv.conftest import DEMO_ARTIFACT
    with client_for(settings, cv_engine="yolov8", cv_model_path=DEMO_ARTIFACT, cv_device="cuda") as c:
        h = c.get("/api/health").json()
        body = run(c, png((10, 200, 30))).json()
    assert h["pipeline_modules"]["cv_engine"]["error"].startswith(("DEVICE_UNAVAILABLE", "MODEL_NOT_FOUND"))
    assert body["status"] == "REQUIRES_REVIEW" and body["decision"] is None
