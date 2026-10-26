import json
import math
from pathlib import Path
import pytest
import jsonschema

import sys
# Ensure repository root is on sys.path so module_c is importable
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from module_c.models import validate_request as module_c_validate_request
from app.models.inspection import Inspection
from app.services.report_service import (
    UnsupportedDefectClassError,
    build_module_c_rca_request,
)
from tests.conftest import FakeDetector, det, predict


SCHEMA_PATH = repo_root / "schemas" / "rca_request.schema.json"


@pytest.fixture
def rca_schema():
    assert SCHEMA_PATH.is_file(), f"Missing schema at {SCHEMA_PATH}"
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _make_inspection(
    *,
    id="insp-test-123",
    site="Site-A",
    image_sha256="a1b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef",
    image_width=640,
    image_height=480,
    model_version="yolo-v8-industrial-1.0",
    model_sha256="deadbeef" * 8,
    policy_version=1,
    confidence_threshold=0.5,
    threshold_overridden=False,
    ai_decision="FAIL",
    final_decision="FAIL",
    defects_json="[]",
):
    insp = Inspection(
        id=id,
        user_id="user-123",
        site=site,
        image_sha256=image_sha256,
        image_width=image_width,
        image_height=image_height,
        model_status="ready",
        model_name="yolov8",
        model_version=model_version,
        model_sha256=model_sha256,
        confidence_threshold=confidence_threshold,
        threshold_overridden=threshold_overridden,
        policy_version=policy_version,
        ai_decision=ai_decision,
        ai_decision_reasons="[]",
        final_decision=final_decision,
        defect_count=0,
        defects_json=defects_json,
        processing_time_ms=12.5,
        inference_time_ms=8.2,
    )
    return insp


# ==============================================================================
# Unit tests for build_module_c_rca_request
# ==============================================================================

def test_valid_inspection_multiple_defects(rca_schema):
    defects = [
        {"class_name": "crack", "confidence": 0.92, "bbox": {"x1": 50, "y1": 60, "x2": 150, "y2": 200}},
        {"class_name": "scratch", "confidence": 0.81, "bbox": {"x1": 200, "y1": 100, "x2": 350, "y2": 120}},
    ]
    insp = _make_inspection(defects_json=json.dumps(defects), ai_decision="FAIL", final_decision="FAIL")
    payload = build_module_c_rca_request(insp, supported_classes=["crack", "scratch", "dent", "corrosion"])

    # 1. Validate structure
    assert payload["inspection_id"] == "insp-test-123"
    assert payload["image_id"] == insp.image_sha256
    assert payload["model_version"] == "yolo-v8-industrial-1.0"
    assert payload["status"] == "defect_detected"
    assert payload["image_width"] == 640
    assert payload["image_height"] == 480
    assert len(payload["defects"]) == 2
    assert payload["defects"][0] == {"class": "crack", "confidence": 0.92, "bbox": [50.0, 60.0, 150.0, 200.0]}
    assert payload["defects"][1] == {"class": "scratch", "confidence": 0.81, "bbox": [200.0, 100.0, 350.0, 120.0]}
    assert "site=" in payload["inspection_context"]

    # 2. Strict rejection of unexpected top-level fields
    forbidden_keys = {"site", "timestamp", "model_sha256", "policy_version", "confidence_threshold", "ai_decision", "final_decision"}
    assert not (set(payload.keys()) & forbidden_keys)

    # 3. JSON Schema cross-validation
    jsonschema.validate(instance=payload, schema=rca_schema)

    # 4. Module C validate_request cross-validation
    rca_req = module_c_validate_request(payload, supported_classes=["crack", "scratch", "dent", "corrosion"], max_defects=20)
    assert rca_req.inspection_id == "insp-test-123"
    assert rca_req.image_id == insp.image_sha256
    assert len(rca_req.defects) == 2


def test_empty_defects_allowed(rca_schema):
    insp = _make_inspection(defects_json="[]", ai_decision="PASS", final_decision="PASS")
    payload = build_module_c_rca_request(insp)
    assert payload["defects"] == []
    assert payload["status"] == "nominal"

    # Schema & Module C validation
    jsonschema.validate(instance=payload, schema=rca_schema)
    rca_req = module_c_validate_request(payload, supported_classes=["crack", "scratch", "dent", "corrosion"], max_defects=20)
    assert len(rca_req.defects) == 0


def test_missing_or_invalid_image_hash():
    insp_empty = _make_inspection(image_sha256="")
    with pytest.raises(ValueError, match="image_sha256 is missing or invalid"):
        build_module_c_rca_request(insp_empty)

    insp_invalid = _make_inspection(image_sha256="!invalid_hash#$")
    with pytest.raises(ValueError, match="image_sha256 is missing or invalid"):
        build_module_c_rca_request(insp_invalid)


def test_missing_or_invalid_model_version():
    insp_empty = _make_inspection(model_version="")
    with pytest.raises(ValueError, match="model_version is missing or invalid"):
        build_module_c_rca_request(insp_empty)

    insp_none = _make_inspection(model_version=None)
    with pytest.raises(ValueError, match="model_version is missing or invalid"):
        build_module_c_rca_request(insp_none)


def test_unsupported_classes_transparently_fails():
    defects = [
        {"class_name": "crack", "confidence": 0.9, "bbox": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}},
        {"class_name": "unsupported_foreign_particle", "confidence": 0.85, "bbox": {"x1": 30, "y1": 30, "x2": 40, "y2": 40}},
    ]
    insp = _make_inspection(defects_json=json.dumps(defects))

    with pytest.raises(UnsupportedDefectClassError) as exc_info:
        build_module_c_rca_request(insp, supported_classes=["crack", "scratch"])

    assert "unsupported_foreign_particle" in exc_info.value.unsupported_classes
    assert exc_info.value.supported_classes == ["crack", "scratch"]


@pytest.mark.parametrize(
    "bad_bbox,match_err",
    [
        ({"x1": 10, "y1": 10, "x2": 5, "y2": 20}, "positive width and height"),  # reversed x
        ({"x1": 10, "y1": 10, "x2": 20, "y2": 5}, "positive width and height"),  # reversed y
        ({"x1": -1, "y1": 10, "x2": 20, "y2": 20}, "non-negative"),             # negative coordinate
        ({"x1": 10, "y1": 10, "x2": 700, "y2": 20}, "exceeds image size"),      # exceeds width 640
        ({"x1": 10, "y1": 10, "x2": 20, "y2": 500}, "exceeds image size"),      # exceeds height 480
        ({"x1": float("nan"), "y1": 10, "x2": 20, "y2": 20}, "finite numbers"),  # NaN
        ({"x1": 10, "y1": 10, "x2": float("inf"), "y2": 20}, "finite numbers"), # Inf
        ({"x1": 10, "y1": 10, "x2": 20}, "missing coordinate 'y2'"),            # missing key
        ([10, 10, 20], "must be 4 coordinates"),                                 # list wrong length
    ],
)
def test_malformed_bounding_boxes(bad_bbox, match_err):
    defects = [{"class_name": "crack", "confidence": 0.85, "bbox": bad_bbox}]
    insp = _make_inspection(defects_json=json.dumps(defects))
    with pytest.raises(ValueError, match=match_err):
        build_module_c_rca_request(insp)


@pytest.mark.parametrize(
    "bad_conf",
    [-0.1, 1.05, float("nan"), float("inf"), True, False, None, "high"],
)
def test_invalid_confidence_values(bad_conf):
    defects = [{"class_name": "crack", "confidence": bad_conf, "bbox": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}}]
    insp = _make_inspection(defects_json=json.dumps(defects))
    with pytest.raises(ValueError, match="invalid confidence"):
        build_module_c_rca_request(insp)


def test_status_mapping_and_override_tracking():
    # 1. PASS -> nominal
    insp_pass = _make_inspection(ai_decision="PASS", final_decision="PASS")
    p_pass = build_module_c_rca_request(insp_pass)
    assert p_pass["status"] == "nominal"

    # 2. FAIL -> defect_detected
    insp_fail = _make_inspection(ai_decision="FAIL", final_decision="FAIL")
    p_fail = build_module_c_rca_request(insp_fail)
    assert p_fail["status"] == "defect_detected"

    # 3. REVIEW -> review_required
    insp_rev = _make_inspection(ai_decision="REVIEW", final_decision="REVIEW")
    p_rev = build_module_c_rca_request(insp_rev)
    assert p_rev["status"] == "review_required"

    # 4. Human Override: ai=FAIL, final=PASS
    insp_ov = _make_inspection(ai_decision="FAIL", final_decision="PASS")
    p_ov = build_module_c_rca_request(insp_ov)
    assert p_ov["status"] == "nominal"  # follows authoritative final decision
    assert "ai_decision=FAIL" in p_ov["inspection_context"]
    assert "final_decision=PASS" in p_ov["inspection_context"]


# ==============================================================================
# Integration tests for GET /inspections/{inspection_id}/rca-request endpoint
# ==============================================================================

def test_rca_request_endpoint_success(client, headers, use_detector, rca_schema):
    # Default image_bytes() in conftest is 64x48
    use_detector(FakeDetector([det("crack", 0.88, (10, 10, 40, 30))]))
    r = predict(client, headers["op"])
    assert r.status_code == 201
    iid = r.json()["inspection_id"]

    resp = client.get(f"/inspections/{iid}/rca-request", headers=headers["op"])
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["inspection_id"] == iid
    assert data["defects"][0]["class"] == "crack"
    assert data["defects"][0]["confidence"] == 0.88
    assert data["defects"][0]["bbox"] == [10.0, 10.0, 40.0, 30.0]

    # Validate against JSON schema and Module C validate_request
    jsonschema.validate(instance=data, schema=rca_schema)
    module_c_validate_request(data, supported_classes=["crack", "scratch", "dent", "corrosion"], max_defects=20)


def test_rca_request_endpoint_unsupported_class_returns_422(client, headers, use_detector):
    # 'porosity' is not in default settings.rca_supported_classes ("scratch,crack,dent,corrosion")
    # Bbox (10, 10, 40, 30) fits within 64x48 image
    use_detector(FakeDetector([det("porosity", 0.85, (10, 10, 40, 30))]))
    r = predict(client, headers["op"])
    assert r.status_code == 201
    iid = r.json()["inspection_id"]

    resp = client.get(f"/inspections/{iid}/rca-request", headers=headers["op"])
    assert resp.status_code == 422
    err = resp.json()["detail"]
    assert "unsupported_classes" in err
    assert "porosity" in err["unsupported_classes"]


def test_rca_request_endpoint_auth_and_scoping(client, headers, use_detector):
    use_detector(FakeDetector([det("crack", 0.9, (5, 5, 30, 25))]))
    iid = predict(client, headers["op"]).json()["inspection_id"]

    # 1. Unauthenticated -> 401
    assert client.get(f"/inspections/{iid}/rca-request").status_code == 401

    # 2. Operator from another site -> 404 (IDOR guard)
    assert client.get(f"/inspections/{iid}/rca-request", headers=headers["op_b"]).status_code == 404

    # 3. Manager from same site -> 200
    assert client.get(f"/inspections/{iid}/rca-request", headers=headers["mgr"]).status_code == 200

    # 4. Admin -> 200
    assert client.get(f"/inspections/{iid}/rca-request", headers=headers["admin"]).status_code == 200
