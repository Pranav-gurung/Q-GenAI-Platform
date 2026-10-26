import json
import time

import jwt
from sqlalchemy import select

from app.config import get_settings
from app.models.audit_event import AuditEvent
from app.models.user import Role
from app.services.authentication import hash_password, verify_password
from tests.conftest import PASSWORD, login, make_user


def test_password_hashing_uses_argon2_and_verifies():
    h = hash_password("Correct-horse-42")
    assert h.startswith("$argon2id$")
    assert "Correct-horse-42" not in h
    assert verify_password("Correct-horse-42", h)
    assert not verify_password("wrong-password-42", h)
    assert hash_password("Correct-horse-42") != h  # random salt


def test_valid_login_returns_token_and_me(client, users):
    r = client.post("/auth/login", data={"username": "operator1", "password": PASSWORD})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["role"] == "operator"
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}).json()
    assert me["username"] == "operator1"
    assert "inspection:run" in me["permissions"] and "user:manage" not in me["permissions"]


def test_invalid_password_and_unknown_user_get_same_generic_error(client, users, db):
    r1 = client.post("/auth/login", data={"username": "operator1", "password": "WrongPass123"})
    r2 = client.post("/auth/login", data={"username": "nobody", "password": "WrongPass123"})
    assert r1.status_code == r2.status_code == 401
    assert r1.json()["detail"] == r2.json()["detail"] == "Incorrect username or password."
    events = db.execute(select(AuditEvent).where(AuditEvent.action == "auth.login_failed")).scalars().all()
    assert len(events) == 2
    for e in events:  # credentials are never logged
        assert "WrongPass123" not in e.metadata_json
        assert json.loads(e.metadata_json)["username_attempted"] in {"operator1", "nobody"}


def test_successful_login_is_audited(client, users, db):
    login(client, "operator1")
    ev = db.execute(select(AuditEvent).where(AuditEvent.action == "auth.login_success")).scalar_one()
    assert ev.actor_id == users["op"].id and ev.actor_role == "operator"


def test_missing_token_is_rejected(client):
    assert client.get("/auth/me").status_code == 401


def test_expired_token_is_rejected(client, users):
    s = get_settings()
    now = int(time.time())
    token = jwt.encode({"sub": users["op"].id, "iss": "q-genai", "iat": now - 3600, "nbf": now - 3600,
                        "exp": now - 60, "jti": "x1", "ver": 0}, s.jwt_secret_key, algorithm="HS256")
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401 and "expired" in r.json()["detail"].lower()


def test_forged_and_tampered_tokens_are_rejected(client, users):
    now = int(time.time())
    claims = {"sub": users["admin"].id, "iss": "q-genai", "iat": now, "exp": now + 600, "jti": "x2", "ver": 0}
    forged = jwt.encode(claims, "attacker-chosen-secret-key-0123456789", algorithm="HS256")
    none_alg = jwt.encode(claims, None, algorithm="none")
    good = login(client, "operator1")["Authorization"].split()[1]
    tampered = good[:-4] + ("AAAA" if not good.endswith("AAAA") else "BBBB")
    for tok in (forged, none_alg, tampered, "not-a-jwt"):
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {tok}"}).status_code == 401


def test_logout_revokes_token(client, users):
    h = login(client, "operator1")
    assert client.post("/auth/logout", headers=h).status_code == 200
    assert client.get("/auth/me", headers=h).status_code == 401


def test_disabled_user_cannot_login_and_old_token_stops_working(client, headers, users):
    op_headers = headers["op"]
    r = client.patch(f"/admin/users/{users['op'].id}", headers=headers["admin"], json={"is_active": False})
    assert r.status_code == 200
    assert client.get("/auth/me", headers=op_headers).status_code == 401
    assert client.post("/auth/login", data={"username": "operator1", "password": PASSWORD}).status_code == 401


def test_role_change_invalidates_existing_tokens(client, headers, users):
    old = headers["op"]
    client.patch(f"/admin/users/{users['op'].id}", headers=headers["admin"], json={"role": "quality_manager"})
    assert client.get("/auth/me", headers=old).status_code == 401
    assert login(client, "operator1")  # new token works and carries the new role
    assert client.get("/auth/me", headers=login(client, "operator1")).json()["role"] == "quality_manager"


def test_login_rate_limit(client, users):
    for _ in range(5):
        assert client.post("/auth/login", data={"username": "operator1", "password": "bad-pass-123"}).status_code == 401
    r = client.post("/auth/login", data={"username": "operator1", "password": PASSWORD})
    assert r.status_code == 429  # blocked even with the correct password
    assert int(r.headers["Retry-After"]) > 0


def test_inactive_user_created_directly_cannot_login(client):
    from app.database import SessionLocal
    u = make_user("ghost", Role.OPERATOR)
    with SessionLocal() as s:
        s.get(type(u), u.id).is_active = False
        s.commit()
    assert client.post("/auth/login", data={"username": "ghost", "password": PASSWORD}).status_code == 401
