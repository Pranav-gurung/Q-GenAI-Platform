
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.detector import (
    Detection,
    DetectorManager,
    ModelUnavailableError,
    YoloDetector,
    looks_like_coco,
    resolve_weights_path,
)


def test_detection_to_dict_rounds_values():
    detection = Detection(
        class_id=1,
        class_name="defect",
        confidence=0.987654,
        x1=1.26,
        y1=2.75,
        x2=10.14,
        y2=20.99,
    )

    result = detection.to_dict()

    assert result["confidence"] == 0.9877
    assert result["bbox"] == {
        "x1": 1.3,
        "y1": 2.8,
        "x2": 10.1,
        "y2": 21.0,
    }


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "../model.pt",
        "..\\model.pt",
        ".hidden.pt",
        "folder/model.pt",
        "model.txt",
    ],
)
def test_resolve_weights_path_rejects_invalid_names(tmp_path, filename):
    with pytest.raises(ValueError):
        resolve_weights_path(tmp_path, filename)


def test_resolve_weights_path_accepts_supported_extension(tmp_path):
    result = resolve_weights_path(tmp_path, "defects.pt")
    assert result == (tmp_path / "defects.pt").resolve()


def test_load_missing_weights_returns_unavailable(tmp_path):
    detector = YoloDetector(tmp_path, "missing.pt", "v1")

    info = detector.load()

    assert info.status == "unavailable"
    assert detector._model is None


def test_load_corrupt_weights_returns_error(tmp_path):
    weights = tmp_path / "corrupt.pt"
    weights.write_bytes(b"not a real model")

    detector = YoloDetector(tmp_path, weights.name, "v1")

    with patch("ultralytics.YOLO", side_effect=RuntimeError("bad model")):
        info = detector.load()

    assert info.status == "error"
    assert detector._model is None
    assert "RuntimeError" in info.message


def test_predict_without_loaded_model_raises(tmp_path):
    detector = YoloDetector(tmp_path, "missing.pt", "v1")

    with pytest.raises(ModelUnavailableError):
        detector.predict(np.zeros((32, 32, 3), dtype=np.uint8), 0.5)


def test_looks_like_coco_rejects_non_coco_names():
    assert not looks_like_coco(["defect", "scratch"])
    assert not looks_like_coco(["person", "toothbrush"])


def test_manager_keeps_previous_model_on_failed_reload(tmp_path):
    manager = DetectorManager()

    previous = MagicMock()
    manager._current = previous

    info = manager.load(
        models_dir=tmp_path,
        weights_file="missing.pt",
        version="v2",
        device="cpu",
        imgsz=640,
        approved=False,
        keep_previous_on_failure=True,
    )

    assert info.status == "unavailable"
    assert manager.current is previous


def test_manager_replaces_previous_model_on_failed_reload_by_default(tmp_path):
    manager = DetectorManager()
    previous = MagicMock()
    manager._current = previous

    info = manager.load(
        models_dir=tmp_path,
        weights_file="missing.pt",
        version="v2",
        device="cpu",
        imgsz=640,
        approved=False,
    )

    assert info.status == "unavailable"
    assert manager.current is not previous
