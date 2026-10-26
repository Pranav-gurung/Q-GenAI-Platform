import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.utils import corrupt_png, image_bytes


def create(client, payload):
    r = client.post("/api/inspections", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def upload(client, inspection_id, data=None, name="product.png", ctype="image/png"):
    data = image_bytes() if data is None else data
    return client.post(f"/api/inspections/{inspection_id}/image", files={"file": (name, data, ctype)})


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["success"] is False and body["error"]["code"] == code
    assert "Traceback" not in r.text
    return body["error"]


class TestCreate:
    def test_create_success(self, client, valid_payload):
        body = create(client, valid_payload)
        assert body["inspection_id"].startswith("INS-")
        assert body["status"] == "CREATED"
        assert body["stage"] == "PACKING"
        assert body["image"] is None and body["decision"] is None
        assert body["schema_version"] == "1.0"

    def test_invalid_stage(self, client, valid_payload):
        err = assert_error(client.post("/api/inspections", json={**valid_payload, "stage": "TELEPORT"}),
                           422, "VALIDATION_ERROR")
        assert err["details"][0]["field"] == "body.stage"

    def test_missing_fields(self, client):
        assert_error(client.post("/api/inspections", json={"stage": "PACKING"}), 422, "VALIDATION_ERROR")

    def test_malformed_json(self, client):
        r = client.post("/api/inspections", content=b"{not json", headers={"content-type": "application/json"})
        assert_error(r, 422, "VALIDATION_ERROR")

    def test_invalid_id_format(self, client, valid_payload):
        assert_error(client.post("/api/inspections", json={**valid_payload, "store_id": "../etc"}),
                     422, "VALIDATION_ERROR")

    def test_unknown_store_and_sku(self, client, valid_payload):
        assert_error(client.post("/api/inspections", json={**valid_payload, "store_id": "DS-NOPE"}),
                     422, "UNKNOWN_STORE")
        assert_error(client.post("/api/inspections", json={**valid_payload, "sku_id": "SKU-NOPE"}),
                     422, "UNKNOWN_SKU")


class TestRetrieve:
    def test_get(self, client, valid_payload):
        created = create(client, valid_payload)
        r = client.get(f"/api/inspections/{created['inspection_id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "CREATED" and body["result"] is None
        assert body["history"][0]["to_state"] == "CREATED"

    def test_not_found(self, client):
        assert_error(client.get("/api/inspections/INS-0000000000000000"), 404, "INSPECTION_NOT_FOUND")

    def test_invalid_id(self, client):
        assert_error(client.get("/api/inspections/not-an-id"), 422, "VALIDATION_ERROR")

    def test_list_and_filter(self, client, valid_payload):
        create(client, valid_payload)
        create(client, {**valid_payload, "stage": "RECEIVING", "store_id": "DS-DEL-001"})
        all_ = client.get("/api/inspections").json()
        assert all_["total"] == 2 and len(all_["items"]) == 2
        only = client.get("/api/inspections", params={"stage": "RECEIVING"}).json()
        assert only["total"] == 1 and only["items"][0]["store_id"] == "DS-DEL-001"
        assert client.get("/api/inspections", params={"status": "COMPLETED"}).json()["total"] == 0
        assert_error(client.get("/api/inspections", params={"limit": 1000}), 422, "VALIDATION_ERROR")


class TestImageAndPipeline:
    def test_upload_runs_pipeline_with_stub_modules(self, client, valid_payload, settings):
        ins = create(client, valid_payload)
        r = upload(client, ins["inspection_id"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "REQUIRES_REVIEW"
        assert body["decision"] is None
        assert body["model_versions"] == {}
        assert body["image"]["width"] == 64 and body["image"]["mime_type"] == "image/png"
        # stored via the storage abstraction
        assert (settings.storage_path / body["image"]["storage_reference"]).is_file()
        # no fake AI output anywhere
        result = body["result"]
        assert result["cv"]["status"] == "NOT_IMPLEMENTED"
        assert result["cv"]["detections"] == [] and result["cv"]["overall_confidence"] is None
        assert result["decision"]["status"] == "SKIPPED" and result["decision"]["outcome"] is None
        transitions = [(h["from_state"], h["to_state"]) for h in body["history"]
                       if h["event_type"] == "STATE_TRANSITION"]
        assert transitions == [(None, "CREATED"), ("CREATED", "IMAGE_RECEIVED"),
                               ("IMAGE_RECEIVED", "QUALITY_CHECK"), ("QUALITY_CHECK", "CV_PROCESSING"),
                               ("CV_PROCESSING", "DECISION"), ("DECISION", "PIPELINE_ANALYSIS"),
                               ("PIPELINE_ANALYSIS", "RCA_PROCESSING"), ("RCA_PROCESSING", "ACTION_GENERATED"),
                               ("ACTION_GENERATED", "REQUIRES_REVIEW")]
        # persisted: a fresh GET returns the same state
        again = client.get(f"/api/inspections/{ins['inspection_id']}").json()
        assert again["status"] == "REQUIRES_REVIEW" and again["result"]["result_id"] == result["result_id"]

    @pytest.mark.parametrize("data,name,ctype,code", [
        (image_bytes(), "product.gif", "image/gif", "UNSUPPORTED_IMAGE_TYPE"),
        (b"hello", "notes.txt", "text/plain", "UNSUPPORTED_IMAGE_TYPE"),
        (corrupt_png(), "product.png", "image/png", "CORRUPT_IMAGE"),
        (b"", "product.png", "image/png", "EMPTY_FILE"),
        (b"\x89PNG" + b"0" * (2 * 1024 * 1024), "big.png", "image/png", "IMAGE_TOO_LARGE"),  # limit is 1 MB in tests
    ])
    def test_invalid_images_rejected(self, client, valid_payload, data, name, ctype, code):
        ins = create(client, valid_payload)
        assert_error(upload(client, ins["inspection_id"], data, name, ctype), 422, code)
        # state unchanged, nothing stored
        assert client.get(f"/api/inspections/{ins['inspection_id']}").json()["status"] == "CREATED"

    def test_missing_file(self, client, valid_payload):
        ins = create(client, valid_payload)
        assert_error(client.post(f"/api/inspections/{ins['inspection_id']}/image"), 422, "VALIDATION_ERROR")

    def test_upload_to_unknown_inspection(self, client):
        assert_error(upload(client, "INS-0000000000000000"), 404, "INSPECTION_NOT_FOUND")

    def test_upload_not_allowed_after_completion(self, client, valid_payload):
        ins = create(client, valid_payload)
        upload(client, ins["inspection_id"])
        client.post(f"/api/inspections/{ins['inspection_id']}/feedback",
                    json={"operator_id": "OP-17", "outcome": "PASS"})
        err = assert_error(upload(client, ins["inspection_id"]), 409, "INVALID_STATE_TRANSITION")
        assert err["details"] == {"from": "COMPLETED", "to": "IMAGE_RECEIVED", "allowed": []}

    def test_retake_from_requires_review(self, client, valid_payload):
        ins = create(client, valid_payload)
        upload(client, ins["inspection_id"])
        r = upload(client, ins["inspection_id"], image_bytes("JPEG"), "retake.jpg", "image/jpeg")
        assert r.status_code == 200
        assert r.json()["image"]["mime_type"] == "image/jpeg"
        assert r.json()["status"] == "REQUIRES_REVIEW"


class TestFeedback:
    def test_feedback_resolves_review(self, client, valid_payload):
        ins = create(client, valid_payload)
        upload(client, ins["inspection_id"])
        r = client.post(f"/api/inspections/{ins['inspection_id']}/feedback", json={
            "operator_id": "OP-17", "outcome": "FAIL", "defect_types": ["LEAKING_PACKAGE"],
            "suspected_stage": "PICKING", "comment": "Leaking at seam",
            "corrected_detections": [{"defect_type": "LEAKING_PACKAGE",
                                      "bbox": {"x_min": 1, "y_min": 2, "x_max": 30, "y_max": 40}}]})
        assert r.status_code == 201, r.text
        fb = r.json()
        assert fb["feedback_id"].startswith("FB-")
        assert fb["system_outcome"] is None and fb["is_correction"] is False
        assert fb["learning_status"] == "PENDING"
        assert fb["corrected_detections"][0]["bbox"]["x_max"] == 30
        insp = client.get(f"/api/inspections/{ins['inspection_id']}").json()
        assert insp["status"] == "COMPLETED" and insp["decision"] == "FAIL"
        assert insp["history"][-1]["message"].startswith("Resolved by operator review")

    def test_feedback_before_pipeline_rejected(self, client, valid_payload):
        ins = create(client, valid_payload)
        assert_error(client.post(f"/api/inspections/{ins['inspection_id']}/feedback",
                                 json={"operator_id": "OP-17", "outcome": "PASS"}), 409, "FEEDBACK_NOT_ALLOWED")

    def test_invalid_feedback(self, client, valid_payload):
        ins = create(client, valid_payload)
        upload(client, ins["inspection_id"])
        assert_error(client.post(f"/api/inspections/{ins['inspection_id']}/feedback",
                                 json={"operator_id": "OP-17", "outcome": "MAYBE"}), 422, "VALIDATION_ERROR")


class TestErrorsAndAuth:
    def test_unhandled_errors_are_generic(self, app, valid_payload, monkeypatch):
        from backend.services import inspection_service

        def boom(*a, **k):
            raise RuntimeError("db password is hunter2")
        monkeypatch.setattr(inspection_service.InspectionService, "create", boom)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/inspections", json=valid_payload)
        err = assert_error(r, 500, "INTERNAL_ERROR")
        assert "hunter2" not in r.text and err["request_id"]

    def test_api_key_enforced_when_configured(self, settings, valid_payload):
        from backend.db.base import Base
        from backend.services.reference_service import seed_sample_data
        app = create_app(settings.model_copy(update={"api_key": "s3cret"}))
        Base.metadata.create_all(app.state.engine)
        with app.state.session_factory() as s:
            seed_sample_data(s, settings.sample_data_path)
        with TestClient(app) as c:
            assert_error(c.post("/api/inspections", json=valid_payload), 401, "UNAUTHORIZED")
            assert_error(c.post("/api/inspections", json=valid_payload, headers={"X-API-Key": "wrong"}),
                         401, "UNAUTHORIZED")
            assert c.post("/api/inspections", json=valid_payload, headers={"X-API-Key": "s3cret"}).status_code == 201
            assert c.get("/api/health").status_code == 200  # reads stay open
