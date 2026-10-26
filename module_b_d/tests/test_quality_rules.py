from app.services.detector import ModelInfo
from app.services.inspection_rules import FAIL, PASS, REVIEW, PolicyRules, evaluate

LOADED = ModelInfo(status="loaded", message="", approved=True)


def d(name, conf):
    return {"class_id": 0, "class_name": name, "confidence": conf,
            "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}


def rules(**kw):
    base = dict(version=1, fail_confidence=0.6)
    base.update(kw)
    return PolicyRules(**base)


def test_no_detections_pass():
    assert evaluate([], rules(), LOADED).status == PASS


def test_confident_defect_fails():
    dec = evaluate([d("crack", 0.9)], rules(), LOADED)
    assert dec.status == FAIL and dec.disqualifying_count == 1 and "crack" in dec.reasons[0]


def test_low_confidence_detection_needs_review():
    assert evaluate([d("crack", 0.4)], rules(), LOADED).status == REVIEW


def test_fail_classes_restrict_disqualifying_defects():
    r = rules(fail_classes=frozenset({"crack"}))
    assert evaluate([d("crack", 0.9)], r, LOADED).status == FAIL
    assert evaluate([d("scratch", 0.95)], r, LOADED).status == REVIEW  # detected but not a fail class


def test_ignored_classes_are_skipped():
    r = rules(ignored_classes=frozenset({"dust"}))
    dec = evaluate([d("dust", 0.99)], r, LOADED)
    assert dec.status == PASS and any("ignored" in x for x in dec.reasons)


def test_max_allowed_defects():
    r = rules(max_allowed_defects=1)
    assert evaluate([d("scratch", 0.9)], r, LOADED).status == PASS
    assert evaluate([d("scratch", 0.9), d("scratch", 0.8)], r, LOADED).status == FAIL


def test_identical_detections_counted_separately():
    r = rules(max_allowed_defects=1)
    same = d("scratch", 0.9)
    assert evaluate([same, dict(same)], r, LOADED).status == FAIL


def test_unavailable_model_review():
    info = ModelInfo(status="unavailable", message="")
    assert evaluate([], rules(), info).status == REVIEW


def test_error_model_review():
    assert evaluate([], rules(), ModelInfo(status="error", message="")).status == REVIEW


def test_generic_model_review_even_with_detections():
    info = ModelInfo(status="loaded", message="", is_generic=True)
    assert evaluate([d("crack", 0.99)], rules(), info).status == REVIEW


def test_unapproved_model_review_when_policy_requires_approval():
    info = ModelInfo(status="loaded", message="", approved=False)
    assert evaluate([], rules(require_approved_model=True), info).status == REVIEW
    assert evaluate([], rules(require_approved_model=False), info).status == PASS


def test_policy_update_validation(client, headers):
    bad = {"default_confidence_threshold": 0.9, "operator_min_threshold": 0.3,
           "operator_max_threshold": 0.8, "fail_confidence": 0.5, "change_reason": "invalid ranges"}
    assert client.put("/quality/policy", headers=headers["admin"], json=bad).status_code == 422
    overlap = {"default_confidence_threshold": 0.5, "operator_min_threshold": 0.3,
               "operator_max_threshold": 0.8, "fail_confidence": 0.5, "fail_classes": ["a"],
               "ignored_classes": ["a"], "change_reason": "overlapping classes"}
    assert client.put("/quality/policy", headers=headers["admin"], json=overlap).status_code == 422


def test_policy_versions_are_kept_and_audited(client, headers):
    body = {"default_confidence_threshold": 0.55, "operator_min_threshold": 0.3,
            "operator_max_threshold": 0.8, "fail_confidence": 0.7, "fail_classes": ["crack"],
            "change_reason": "Customer asked to only reject cracks"}
    new = client.put("/quality/policy", headers=headers["admin"], json=body).json()
    assert new["version"] == 2 and new["fail_classes"] == ["crack"]
    hist = client.get("/quality/policy/history", headers=headers["op"]).json()
    assert [p["version"] for p in hist] == [2, 1]
    events = client.get("/admin/audit-logs?action=policy.updated", headers=headers["admin"]).json()["items"]
    assert events[0]["metadata"]["changes"]["fail_confidence"] == {"old": 0.5, "new": 0.7}
