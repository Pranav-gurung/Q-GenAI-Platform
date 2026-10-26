"""Hash-chain tests: deliberately tamper with the stored log and make sure
verification catches it (and document what it cannot catch on its own)."""
import json

from sqlalchemy import select, text

from app.models.audit_event import AuditEvent
from app.services.audit_service import (
    append_event,
    audited_transaction,
    create_checkpoint,
    sanitize_metadata,
    verify_chain,
)
from app.utils.hashing import GENESIS_HASH, compute_audit_hash
from tests.conftest import FakeDetector, det, predict

KEY = "test-only-checkpoint-key"


def add_events(db, n):
    for i in range(n):
        with audited_transaction(db):
            append_event(db, "test.event", target_type="test", target_id=str(i), metadata={"i": i})


def test_chain_links_and_verifies(db):
    add_events(db, 5)
    events = db.execute(select(AuditEvent).order_by(AuditEvent.seq)).scalars().all()
    assert events[0].prev_hash == GENESIS_HASH
    for prev, cur in zip(events, events[1:]):
        assert cur.prev_hash == prev.hash and cur.seq == prev.seq + 1
    r = verify_chain(db)
    assert r.valid and r.verified_events == r.total_events == len(events)


def test_altered_content_detected(db):
    add_events(db, 5)
    db.execute(text("UPDATE audit_events SET metadata_json = :m WHERE seq = 3"), {"m": '{"i":999}'})
    db.commit()
    r = verify_chain(db)
    assert not r.valid and r.first_failure_seq == 3 and "Content altered" in r.failure_reason
    assert r.verified_events == 2


def test_altered_actor_detected(db):
    add_events(db, 3)
    db.execute(text("UPDATE audit_events SET actor_role = 'admin' WHERE seq = 2"))
    db.commit()
    assert verify_chain(db).first_failure_seq == 2


def test_deleted_event_detected(db):
    add_events(db, 5)
    db.execute(text("DELETE FROM audit_events WHERE seq = 3"))
    db.commit()
    r = verify_chain(db)
    assert not r.valid and r.first_failure_seq == 4 and "Sequence gap" in r.failure_reason


def test_reordered_events_detected(db):
    add_events(db, 5)
    # swap the positions of events 2 and 4
    db.execute(text("UPDATE audit_events SET seq = -2 WHERE seq = 2"))
    db.execute(text("UPDATE audit_events SET seq = 2 WHERE seq = 4"))
    db.execute(text("UPDATE audit_events SET seq = 4 WHERE seq = -2"))
    db.commit()
    r = verify_chain(db)
    assert not r.valid and r.first_failure_seq == 2


def test_rehashed_single_event_breaks_next_link(db):
    add_events(db, 4)
    ev = db.get(AuditEvent, 2)
    ev.metadata_json = '{"i":42}'
    fields = {k: getattr(ev, k) for k in ("seq", "event_id", "timestamp", "actor_id", "actor_role",
                                          "action", "target_type", "target_id", "metadata_json")}
    ev.hash = compute_audit_hash(fields, ev.prev_hash)  # attacker fixes this event's own hash
    db.commit()
    r = verify_chain(db)
    assert not r.valid and r.first_failure_seq == 3 and "Broken link" in r.failure_reason


def _rewrite_whole_chain_from(db, seq):
    """Simulates a privileged attacker who edits an event and recomputes ALL hashes."""
    events = db.execute(select(AuditEvent).order_by(AuditEvent.seq)).scalars().all()
    prev = events[seq - 2].hash if seq > 1 else GENESIS_HASH
    for ev in events[seq - 1:]:
        if ev.seq == seq:
            ev.metadata_json = '{"i":"forged"}'
        fields = {k: getattr(ev, k) for k in ("seq", "event_id", "timestamp", "actor_id", "actor_role",
                                              "action", "target_type", "target_id", "metadata_json")}
        ev.prev_hash = prev
        ev.hash = compute_audit_hash(fields, prev)
        prev = ev.hash
    db.commit()


def test_full_rewrite_is_NOT_detected_without_checkpoint__documented_limitation(db):
    add_events(db, 5)
    _rewrite_whole_chain_from(db, 2)
    assert verify_chain(db).valid  # this is exactly the limitation in docs/SECURITY.md


def test_full_rewrite_detected_with_signed_checkpoint(db):
    add_events(db, 5)
    cp = create_checkpoint(db, KEY)  # stored outside the DB in real use
    _rewrite_whole_chain_from(db, 2)
    r = verify_chain(db, cp, KEY)
    assert not r.valid and "differs from the signed checkpoint" in r.failure_reason


def test_truncation_detected_with_checkpoint(db):
    add_events(db, 5)
    cp = create_checkpoint(db, KEY)
    db.execute(text("DELETE FROM audit_events WHERE seq >= 4"))
    db.commit()
    assert verify_chain(db).valid  # tail truncation alone is invisible...
    r = verify_chain(db, cp, KEY)  # ...but not against a checkpoint
    assert not r.valid and "truncated" in r.failure_reason


def test_forged_checkpoint_rejected(db):
    add_events(db, 3)
    cp = create_checkpoint(db, KEY)
    cp["signature"] = "0" * 64
    r = verify_chain(db, cp, KEY)
    assert not r.valid and "signature is invalid" in r.failure_reason


def test_metadata_sanitizer_redacts_secrets():
    clean = sanitize_metadata({"password": "p@ss", "access_token": "eyJ...", "nested": {"api_key": "k"},
                               "username": "bob", "image_sha256": "abc", "long": "x" * 900})
    assert clean["password"] == clean["access_token"] == clean["nested"]["api_key"] == "[REDACTED]"
    assert clean["username"] == "bob" and clean["image_sha256"] == "abc"
    assert len(clean["long"]) < 600


def test_verify_endpoint_admin_only_and_detects_tampering(client, headers, use_detector, db):
    use_detector(FakeDetector([det()]))
    predict(client, headers["op"])
    assert client.post("/admin/audit-logs/verify", headers=headers["mgr"]).status_code == 403
    assert client.post("/admin/audit-logs/verify", headers=headers["op"]).status_code == 403

    ok = client.post("/admin/audit-logs/verify", headers=headers["admin"]).json()
    assert ok["valid"] is True and ok["first_failure_seq"] is None

    target = db.execute(select(AuditEvent).where(AuditEvent.action == "inspection.result_created")).scalar_one()
    meta = json.loads(target.metadata_json)
    meta["ai_decision"] = "PASS"  # someone tries to hide a FAIL
    db.execute(text("UPDATE audit_events SET metadata_json = :m WHERE seq = :s"),
               {"m": json.dumps(meta, separators=(",", ":"), sort_keys=True), "s": target.seq})
    db.commit()

    bad = client.post("/admin/audit-logs/verify", headers=headers["admin"]).json()
    assert bad["valid"] is False and bad["first_failure_seq"] == target.seq
    # the verification runs themselves are audit-logged
    logged = client.get("/admin/audit-logs?action=audit.verified", headers=headers["admin"]).json()
    assert logged["total"] == 2


def test_checkpoint_endpoint_roundtrip(client, headers):
    cp = client.post("/admin/audit-logs/checkpoint", headers=headers["admin"]).json()
    body = {"checkpoint": {k: cp[k] for k in ("seq", "hash", "created_at", "signature")}}
    r = client.post("/admin/audit-logs/verify", headers=headers["admin"], json=body).json()
    assert r["valid"] and r["checkpoint_checked"]


def test_audit_events_never_contain_tokens_or_passwords(client, headers, users, use_detector, db):
    use_detector(FakeDetector([det()]))
    predict(client, headers["op"])
    client.post("/admin/users", headers=headers["admin"],
                json={"username": "newbie", "password": "VerySecret123", "role": "operator"})
    client.patch(f"/admin/users/{users['op2'].id}", headers=headers["admin"], json={"password": "AnotherSecret456"})
    client.post("/auth/login", data={"username": "operator1", "password": "TypoPassw0rd"})
    dump = " ".join(e.metadata_json for e in db.execute(select(AuditEvent)).scalars())
    for secret in ("VerySecret123", "AnotherSecret456", "TypoPassw0rd", "TestPassw0rd!", "$argon2",
                   headers["op"]["Authorization"].split()[1]):
        assert secret not in dump
