import pytest

from tests.conftest import FakeDetector, det, predict

POLICY = {"default_confidence_threshold": 0.5, "operator_min_threshold": 0.3,
          "operator_max_threshold": 0.8, "fail_confidence": 0.6, "fail_classes": [],
          "ignored_classes": [], "max_allowed_defects": 0, "require_approved_model": False,
          "change_reason": "Tighter policy for test"}

PROTECTED = [
    ("get", "/auth/me"), ("post", "/auth/logout"), ("get", "/model-info"),
    ("get", "/inspections"), ("get", "/inspections/abc"), ("get", "/inspections/abc/report"),
    ("get", "/inspections/abc/annotated-image"), ("get", "/inspections/abc/rca-payload"),
    ("post", "/predict"), ("get", "/quality/policy"), ("put", "/quality/policy"),
    ("post", "/inspections/abc/override"), ("get", "/quality/review-queue"),
    ("get", "/quality/audit-logs"), ("get", "/admin/users"), ("post", "/admin/users"),
    ("patch", "/admin/users/abc"), ("get", "/admin/audit-logs"), ("post", "/admin/audit-logs/verify"),
    ("post", "/admin/audit-logs/checkpoint"), ("get", "/admin/models"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_protected_endpoints_require_authentication(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_health_is_public(client):
    assert client.get("/health").status_code == 200


ADMIN_ONLY = [
    ("get", "/admin/users", None), ("post", "/admin/users", {"username": "x_user", "password": "abcdefgh12",
                                                             "role": "operator"}),
    ("get", "/admin/audit-logs", None), ("post", "/admin/audit-logs/verify", None),
    ("get", "/admin/models", None), ("put", "/quality/policy", POLICY),
]


@pytest.mark.parametrize("who", ["op", "mgr"])
@pytest.mark.parametrize("method,path,body", ADMIN_ONLY)
def test_non_admins_cannot_use_admin_endpoints(client, headers, who, method, path, body):
    kw = {"json": body} if body else {}
    assert getattr(client, method)(path, headers=headers[who], **kw).status_code == 403


def test_admin_can_manage_users_and_policy(client, headers):
    r = client.post("/admin/users", headers=headers["admin"],
                    json={"username": "newop", "password": "Abcdefgh123", "role": "operator", "site": "plant-a"})
    assert r.status_code == 201 and r.json()["role"] == "operator"
    assert "password" not in r.text and "argon2" not in r.text
    assert client.put("/quality/policy", headers=headers["admin"], json=POLICY).json()["version"] == 2
    assert client.get("/admin/users", headers=headers["admin"]).status_code == 200


def test_operator_cannot_escalate_own_role(client, headers, users):
    r = client.patch(f"/admin/users/{users['op'].id}", headers=headers["op"], json={"role": "admin"})
    assert r.status_code == 403
    assert client.get("/auth/me", headers=headers["op"]).json()["role"] == "operator"


def test_extra_privilege_fields_are_rejected(client, headers):
    r = client.post("/admin/users", headers=headers["admin"],
                    json={"username": "sneaky", "password": "Abcdefgh123", "role": "operator",
                          "is_superuser": True})
    assert r.status_code == 422


def test_weak_password_rejected(client, headers):
    r = client.post("/admin/users", headers=headers["admin"],
                    json={"username": "weak", "password": "short", "role": "operator"})
    assert r.status_code == 422


def test_admin_cannot_demote_or_disable_self(client, headers, users):
    for body in ({"role": "operator"}, {"is_active": False}):
        assert client.patch(f"/admin/users/{users['admin'].id}", headers=headers["admin"], json=body).status_code == 400


def test_admin_does_not_run_inspections_separation_of_duties(client, headers, use_detector):
    use_detector(FakeDetector())
    assert predict(client, headers["admin"]).status_code == 403


def test_manager_permissions(client, headers, users, use_detector):
    use_detector(FakeDetector([det("scratch", 0.9)]))
    insp = predict(client, headers["op"]).json()
    # manager at the same site can see it and the scoped audit log
    assert client.get(f"/inspections/{insp['inspection_id']}", headers=headers["mgr"]).status_code == 200
    assert client.get("/quality/audit-logs", headers=headers["mgr"]).status_code == 200
    assert insp["inspection_id"] in client.get("/quality/review-queue", headers=headers["mgr"]).json()
    # ...and can record a decision
    r = client.post(f"/inspections/{insp['inspection_id']}/override", headers=headers["mgr"],
                    json={"new_decision": "PASS", "reason": "Cosmetic scratch within customer spec"})
    assert r.status_code == 200


def test_operator_cannot_override_or_see_audit(client, headers, use_detector):
    use_detector(FakeDetector([det("scratch", 0.9)]))
    insp = predict(client, headers["op"]).json()
    r = client.post(f"/inspections/{insp['inspection_id']}/override", headers=headers["op"],
                    json={"new_decision": "PASS", "reason": "I think it is fine really"})
    assert r.status_code == 403
    assert client.get("/quality/audit-logs", headers=headers["op"]).status_code == 403


def test_manager_audit_view_is_scoped_to_site(client, headers, users):
    items = client.get("/quality/audit-logs?limit=1000", headers=headers["mgr"]).json()["items"]
    plant_a_ids = {users[k].id for k in ("admin", "op", "op2", "mgr")}
    assert items and all(e["actor_id"] in plant_a_ids for e in items)
    assert all(e["actor_id"] != users["op_b"].id for e in items)
