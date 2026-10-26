"""Optional integration test with REAL YOLO weights (skipped by default).

    QGENAI_TEST_WEIGHTS=models/smoke_test_model.pt python -m pytest tests/test_real_model.py -v

Checks that real weights load, run, and produce schema-valid output through
the API. It asserts nothing about accuracy.
"""
import os
from pathlib import Path

import pytest

from app.services.detector import DetectorManager
from tests.conftest import image_bytes, predict

WEIGHTS = os.environ.get("QGENAI_TEST_WEIGHTS")
pytestmark = pytest.mark.skipif(not WEIGHTS, reason="set QGENAI_TEST_WEIGHTS to run")


def test_real_weights_end_to_end(client, headers, use_detector):
    pytest.importorskip("ultralytics")
    path = Path(WEIGHTS).resolve()
    mgr = DetectorManager()
    info = mgr.load(path.parent, path.name, "integration-test", "cpu", 320, approved=True)
    assert info.status == "loaded", info.message
    assert len(info.sha256) == 64 and info.class_names
    use_detector(mgr.current)

    res = predict(client, headers["op"], data=image_bytes("PNG", (320, 320))).json()
    assert res["model_status"] == "loaded" and res["model_sha256"] == info.sha256
    assert res["inference_time_ms"] > 0
    assert res["inspection_status"] in {"PASS", "FAIL", "REVIEW"}
    for d in res["defects"]:
        assert d["class_name"] in info.class_names and 0 <= d["confidence"] <= 1
