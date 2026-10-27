"""YoloCVEngine with REAL Ultralytics inference (untrained fixture model) + failure modes."""
import io
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("ultralytics")

from backend.domain.enums import ModuleStatus  # noqa: E402
from ml.cv.model import ModelUnavailableError, build_class_map, clear_cache, find_metadata, load_model, resolve_device  # noqa: E402
from ml.cv.postprocess import MalformedPredictionError  # noqa: E402
from ml.cv.taxonomy import load_taxonomy  # noqa: E402
from ml.cv.yolo_engine import YoloCVEngine  # noqa: E402
from tests.cv.conftest import FIXTURE_VERSION, build_fixture_model  # noqa: E402
from tests.cv.helpers import image_input, textured  # noqa: E402

TAX = load_taxonomy()


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


IMG = png(textured((320, 240), seed=1))


def engine(settings, **update):
    return YoloCVEngine(settings.model_copy(update=update))


class TestUnavailable:
    def test_missing_model_is_explicitly_unavailable(self, cv_settings, tmp_path):
        e = engine(cv_settings, cv_model_path=tmp_path / "nope.pt")
        assert not e.available
        h = e.health()
        assert h["available"] is False and h["model_loaded"] is False and h["error"].startswith("MODEL_NOT_FOUND")
        r = e.detect(image_input(IMG))
        assert r.status is ModuleStatus.UNAVAILABLE and r.detections == [] and r.inference is None
        assert r.overall_confidence is None and "MODEL_NOT_FOUND" in r.message

    def test_corrupt_model(self, cv_settings, tmp_path):
        bad = tmp_path / "corrupt.pt"
        bad.write_bytes(b"\x80\x02this is not a checkpoint")
        e = engine(cv_settings, cv_model_path=bad)
        assert e.health()["error"].startswith("MODEL_LOAD_FAILED")
        assert e.detect(image_input(IMG)).status is ModuleStatus.UNAVAILABLE

    def test_taxonomy_mismatch(self, cv_settings, tmp_path):
        w = build_fixture_model(tmp_path / "other.pt", ["scratch", "crack"])
        assert engine(cv_settings, cv_model_path=w).health()["error"].startswith("TAXONOMY_MISMATCH")

    def test_cuda_requested_without_gpu_fails_clearly(self, cv_settings):
        import torch
        if torch.cuda.is_available():
            pytest.skip("machine has CUDA")
        e = engine(cv_settings, cv_device="cuda")
        assert e.health()["error"].startswith("DEVICE_UNAVAILABLE")
        assert e.detect(image_input(IMG)).status is ModuleStatus.UNAVAILABLE


class TestDevice:
    def test_resolution(self):
        import torch
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("auto") == ("cuda:0" if torch.cuda.is_available() else "cpu")
        with pytest.raises(ModelUnavailableError) as exc:
            resolve_device("tpu")
        assert exc.value.code == "INVALID_DEVICE"


class TestModelMetadata:
    def test_version_from_metadata_override_and_hash(self, fixture_model, tmp_path):
        clear_cache()
        assert load_model(fixture_model, "cpu", TAX).info.version == FIXTURE_VERSION
        assert load_model(fixture_model, "cpu", TAX, "pinned-v9").info.version == "pinned-v9"
        bare = build_fixture_model(tmp_path / "x" / "w.pt", TAX.names)
        info = load_model(bare, "cpu", TAX).info
        assert info.version == f"sha256-{info.artifact_sha256[:12]}"
        assert info.artifact.startswith("w.pt@sha256:") and str(tmp_path) not in info.artifact
        assert info.train_image_size == 160 and info.device == "cpu" and "ultralytics" in info.framework
        assert find_metadata(bare) == {}

    def test_model_is_cached(self, fixture_model):
        assert load_model(fixture_model, "cpu", TAX) is load_model(fixture_model, "cpu", TAX)

    def test_class_map_by_name(self):
        m = build_class_map({0: "leakage", 1: "damaged_box"}, TAX)
        assert m[0].name == "leakage" and m[1].defect_type.value == "DAMAGED_BOX"


class TestRealInference:
    def test_untrained_model_at_default_threshold_reports_nothing(self, cv_settings):
        r = engine(cv_settings).detect(image_input(IMG))
        assert r.status is ModuleStatus.COMPLETED and r.is_real
        assert r.detections == [] and r.classification == "NO_DEFECTS_DETECTED" and r.overall_confidence is None
        inf = r.inference
        assert inf.model_version == FIXTURE_VERSION and r.module_version == FIXTURE_VERSION
        assert inf.inference_device == "cpu" and inf.image_size == 160
        assert (inf.image_width, inf.image_height) == (320, 240)
        assert inf.inference_time_ms > 0 and inf.confidence_threshold == 0.35
        assert inf.raw_detection_count == 0

    def test_real_outputs_are_converted(self, cv_settings):
        # threshold 0 keeps the (meaningless but REAL) low-confidence boxes of the untrained net
        r = engine(cv_settings, cv_confidence_threshold=0.0, cv_max_detections=7).detect(image_input(IMG))
        assert r.status is ModuleStatus.COMPLETED and 0 < len(r.detections) <= 7
        assert r.inference.raw_detection_count >= len(r.detections)
        for d in r.detections:
            assert d.model_class in TAX.names and d.severity is not None
            assert 0 <= d.bbox.x_min < d.bbox.x_max <= 320 and 0 <= d.bbox.y_min < d.bbox.y_max <= 240
            assert d.confidence < 0.1  # untrained: nothing confident, and nothing invented
        assert r.overall_confidence == max(d.confidence for d in r.detections)
        assert sum(x.detection_count for x in r.defects) == len(r.detections)

    def test_deterministic(self, cv_settings):
        e = engine(cv_settings, cv_confidence_threshold=0.0)
        a, b = e.detect(image_input(IMG)), e.detect(image_input(IMG))
        assert [d.model_dump() for d in a.detections] == [d.model_dump() for d in b.detections]

    def test_image_size_override(self, cv_settings):
        assert engine(cv_settings, cv_image_size=224).detect(image_input(IMG)).inference.image_size == 224

    def test_health_reports_loaded_model(self, cv_settings):
        h = engine(cv_settings).health()
        assert h["available"] and h["model_loaded"] and h["model_version"] == FIXTURE_VERSION
        assert h["device"] == "cpu" and len(h["classes"]) == 10

    def test_malformed_model_output_raises(self, cv_settings, monkeypatch):
        e = engine(cv_settings)

        class Boxes:
            def __init__(self):
                import torch
                self.xyxy = torch.tensor([[1.0, 2.0, float("nan"), 4.0]])
                self.conf = torch.tensor([0.9])
                self.cls = torch.tensor([1.0])

        class Result:
            boxes = Boxes()
            speed = {}

        monkeypatch.setattr(e, "_predict", lambda bgr: [Result()])
        with pytest.raises(MalformedPredictionError):
            e.detect(image_input(IMG))

    def test_undecodable_bytes_raise(self, cv_settings):
        with pytest.raises(ValueError):
            engine(cv_settings).detect(image_input(b"garbage"))


def test_predict_cli(cv_settings, fixture_model, tmp_path, capsys, monkeypatch):
    from ml.cv import predict
    img = tmp_path / "p.png"
    img.write_bytes(IMG)
    monkeypatch.setenv("CV_MODEL_PATH", str(fixture_model))
    monkeypatch.setenv("CV_DEVICE", "cpu")
    from backend.core.config import get_settings
    get_settings.cache_clear()
    try:
        code = predict.main(["--image", str(img), "--conf", "0", "--save-visualization", str(tmp_path / "v.png")])
    finally:
        get_settings.cache_clear()
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["cv"]["status"] == "COMPLETED" and out["quality"]["acceptable"] is True
    assert out["cv"]["inference"]["model_version"] == FIXTURE_VERSION
    assert Image.open(tmp_path / "v.png").size == (320, 240)


def test_visualize_draws_only_given_detections():
    from backend.schemas.results import BoundingBox, Detection
    from ml.cv.visualize import draw_detections
    base = Image.new("RGB", (100, 80), (255, 255, 255))
    assert np.array_equal(np.asarray(draw_detections(base, [])), np.asarray(base))
    det = Detection(defect_type="LEAKING_PACKAGE", confidence=0.9, severity="HIGH",
                    bbox=BoundingBox(x_min=10, y_min=30, x_max=60, y_max=70))
    out = np.asarray(draw_detections(base, [det]))
    assert (out != 255).any() and (out[75:, 70:] == 255).all()  # pixels outside the box untouched
