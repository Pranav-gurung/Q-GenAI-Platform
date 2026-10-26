import json

from sqlalchemy import select

from app.models.audit_event import AuditEvent
from app.models.inspection import Inspection, InspectionOverride
from app.services.detector import DetectorManager
from tests.conftest import FakeDetector, det, image_bytes, predict


def test_fail_inspection_is_persisted_with_full_provenance(client, headers, users, use_detector, db):
    use_detector(FakeDetector([det("scratch", 0.91, (120, 80, 250, 160)), det("rust", 0.75)]))
    r = predict(client, headers["op"])
    assert r.status_code == 201, r.text
    res = r.json()
    assert res["inspection_status"] == "FAIL" and res["final_decision"] == "FAIL"
    assert res["defect_count"] == 2
    assert res["defects"][0]["bbox"] == {"x1": 120, "y1": 80, "x2": 250, "y2": 160}
    assert res["model_version"] == "test-1" and res["model_sha256"] == "ab" * 32
    assert res["policy_version"] == 1 and res["confidence_threshold"] == 0.5
    assert res["processing_time_ms"] > 0 and res["inference_time_ms"] is not None
    assert len(res["audit_event_ids"]) == 2

    row = db.get(Inspection, res["inspection_id"])
    assert row.user_id == users["op"].id and row.ai_decision == "FAIL"
    assert json.loads(row.defects_json)[0]["class_name"] == "scratch"
    fetched = client.get(f"/inspections/{res['inspection_id']}", headers=headers["op"]).json()
    assert fetched["defects"] == res["defects"]

    actions = [e.action for e in db.execute(select(AuditEvent).order_by(AuditEvent.seq)).scalars()]
    assert actions[-3:] == ["inspection.requested", "inspection.completed", "inspection.result_created"]


def test_zero_detections_pass(client, headers, use_detector):
    use_detector(FakeDetector([]))
    res = predict(client, headers["op"]).json()
    assert res["inspection_status"] == "PASS" and res["defect_count"] == 0 and res["defects"] == []


def test_threshold_filters_detections(client, headers, use_detector):
    use_detector(FakeDetector([det("scratch", 0.45)]))
    res = predict(client, headers["op"], confidence_threshold=0.6).json()
    assert res["defect_count"] == 0 and res["confidence_threshold"] == 0.6


def test_missing_weights_gives_review_and_no_fabricated_detections(client, headers, use_detector, tmp_path):
    mgr = DetectorManager()
    info = mgr.load(tmp_path, "nope.pt", "v0", "cpu", 640, False)
    assert info.status == "unavailable"
    use_detector(mgr.current)
    res = predict(client, headers["op"]).json()
    assert res["model_status"] == "unavailable"
    assert res["inspection_status"] == "REVIEW"
    assert res["defects"] == [] and res["inference_time_ms"] is None


def test_invalid_weights_file_is_handled(tmp_path):
    (tmp_path / "broken.pt").write_bytes(b"this is not a model")
    info = DetectorManager().load(tmp_path, "broken.pt", "v0", "cpu", 640, False)
    assert info.status == "error"
    assert DetectorManager().load(tmp_path, "../etc/passwd", "v0", "cpu", 640, False).status == "error"


def test_generic_coco_model_forces_review(client, headers, use_detector):
    use_detector(FakeDetector([det("scratch", 0.99)], is_generic=True))
    res = predict(client, headers["op"]).json()
    assert res["inspection_status"] == "REVIEW"


def test_operator_cannot_access_another_users_inspection(client, headers, use_detector):
    use_detector(FakeDetector([det()]))
    iid = predict(client, headers["op"]).json()["inspection_id"]
    for path in ("", "/report", "/annotated-image", "/rca-payload"):
        assert client.get(f"/inspections/{iid}{path}", headers=headers["op2"]).status_code == 404
        assert client.get(f"/inspections/{iid}{path}", headers=headers["mgr_b"]).status_code == 404
        assert client.get(f"/inspections/{iid}{path}", headers=headers["op"]).status_code == 200
    listed = client.get("/inspections", headers=headers["op2"]).json()
    assert listed["total"] == 0
    assert client.get("/inspections", headers=headers["admin"]).json()["total"] == 1


def test_report_and_annotated_image(client, headers, use_detector):
    use_detector(FakeDetector([det()]))
    iid = predict(client, headers["op"]).json()["inspection_id"]
    r = client.get(f"/inspections/{iid}/report", headers=headers["op"])
    assert "attachment" in r.headers["content-disposition"]
    rep = r.json()
    assert rep["inspection"]["inspection_id"] == iid
    assert "NOT a certificate of conformity" in rep["disclaimer"]
    assert [e["action"] for e in rep["audit_trail"]] == ["inspection.completed", "inspection.result_created"]
    img = client.get(f"/inspections/{iid}/annotated-image", headers=headers["op"])
    assert img.headers["content-type"] == "image/png" and img.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_rca_payload_contains_no_image_or_user_data(client, headers, use_detector):
    use_detector(FakeDetector([det("crack", 0.8)]))
    iid = predict(client, headers["op"]).json()["inspection_id"]
    p = client.get(f"/inspections/{iid}/rca-payload", headers=headers["op"]).json()
    assert set(p) == {"inspection_id", "timestamp", "site", "model_version", "model_sha256",
                      "policy_version", "confidence_threshold", "ai_decision", "final_decision",
                      "defects", "image_width", "image_height"}
    assert p["defects"][0]["class_name"] == "crack"


def test_operator_threshold_outside_range_is_forbidden(client, headers, use_detector, db):
    use_detector(FakeDetector())
    r = predict(client, headers["op"], confidence_threshold=0.2)
    assert r.status_code == 403
    assert db.execute(select(Inspection)).first() is None
    assert db.execute(select(AuditEvent).where(AuditEvent.action == "inspection.rejected")).first()


def test_manager_threshold_override_requires_reason_and_is_audited(client, headers, use_detector, db):
    use_detector(FakeDetector([det("scratch", 0.25)]))
    assert predict(client, headers["mgr"], confidence_threshold=0.2).status_code == 400
    r = predict(client, headers["mgr"], confidence_threshold=0.2,
                override_reason="Trial of low threshold on new supplier batch")
    assert r.status_code == 201
    res = r.json()
    assert res["threshold_overridden"] is True and res["confidence_threshold"] == 0.2
    ev = db.execute(select(AuditEvent).where(AuditEvent.action == "quality.threshold_override")).scalar_one()
    meta = json.loads(ev.metadata_json)
    assert meta["old_value"] == 0.5 and meta["new_value"] == 0.2 and "supplier" in meta["reason"]
    assert ev.actor_id and ev.target_id == res["inspection_id"]


def test_override_keeps_original_ai_result(client, headers, use_detector, db):
    use_detector(FakeDetector([det("scratch", 0.9)]))
    iid = predict(client, headers["op"]).json()["inspection_id"]
    r = client.post(f"/inspections/{iid}/override", headers=headers["mgr"],
                    json={"new_decision": "PASS", "reason": "Scratch on non-visible surface"})
    res = r.json()
    assert res["final_decision"] == "PASS"
    assert res["inspection_status"] == "FAIL" and res["ai_result"]["inspection_status"] == "FAIL"
    assert res["overrides"][0]["old_decision"] == "FAIL" and res["overrides"][0]["new_decision"] == "PASS"
    ev = db.execute(select(AuditEvent).where(AuditEvent.action == "inspection.manual_override")).scalar_one()
    assert json.loads(ev.metadata_json)["reason"] == "Scratch on non-visible surface"


def test_override_rules(client, headers, use_detector):
    use_detector(FakeDetector([det("scratch", 0.9)]))
    iid = predict(client, headers["op"]).json()["inspection_id"]
    short = client.post(f"/inspections/{iid}/override", headers=headers["mgr"],
                        json={"new_decision": "PASS", "reason": "ok"})
    assert short.status_code == 422  # mandatory reason
    other_site = client.post(f"/inspections/{iid}/override", headers=headers["mgr_b"],
                             json={"new_decision": "PASS", "reason": "Not my site but trying"})
    assert other_site.status_code == 404
    admin = client.post(f"/inspections/{iid}/override", headers=headers["admin"],
                        json={"new_decision": "PASS", "reason": "Admin trying to override"})
    assert admin.status_code == 403
    own = predict(client, headers["mgr"]).json()["inspection_id"]
    r = client.post(f"/inspections/{own}/override", headers=headers["mgr"],
                    json={"new_decision": "PASS", "reason": "Reviewing my own inspection"})
    assert r.status_code == 403  # separation of duties


def test_audit_failure_rolls_back_inspection(client, headers, use_detector, db, monkeypatch):
    import app.services.inspection_service as svc
    from app.config import get_settings
    from app.services.audit_service import AuditWriteError

    use_detector(FakeDetector([det()]))

    def broken(*a, **k):
        raise AuditWriteError("simulated audit store failure")

    annotated = get_settings().storage_dir / "annotated"
    before = set(annotated.glob("*.png"))
    monkeypatch.setattr(svc, "append_event", broken)
    r = predict(client, headers["op"])
    assert r.status_code == 500
    assert "NOT saved" in r.json()["detail"]
    assert db.execute(select(Inspection)).first() is None  # nothing half-recorded
    assert set(annotated.glob("*.png")) == before  # orphan image cleaned up


def test_override_audit_failure_keeps_decision_unchanged(client, headers, use_detector, db, monkeypatch):
    import app.services.inspection_service as svc
    from app.services.audit_service import AuditWriteError

    use_detector(FakeDetector([det()]))
    iid = predict(client, headers["op"]).json()["inspection_id"]

    def broken(*a, **k):
        raise AuditWriteError("simulated")

    monkeypatch.setattr(svc, "append_event", broken)
    r = client.post(f"/inspections/{iid}/override", headers=headers["mgr"],
                    json={"new_decision": "PASS", "reason": "Should not be saved at all"})
    assert r.status_code == 500
    db.expire_all()
    assert db.get(Inspection, iid).final_decision == "FAIL"
    assert db.execute(select(InspectionOverride)).first() is None


def test_jpeg_upload_works(client, headers, use_detector):
    use_detector(FakeDetector())
    r = predict(client, headers["op"], data=image_bytes("JPEG"), filename="part.jpg")
    assert r.status_code == 201 and r.json()["image_width"] == 64
