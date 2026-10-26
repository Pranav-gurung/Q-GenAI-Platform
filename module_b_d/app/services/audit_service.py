"""Section D: SHA-256 hash-chained audit log.

    hash_n = SHA256( "qgenai-audit-v1\\n" || canonical_json({...event_n..., "prev_hash": hash_{n-1}}) )

Threat model (see docs/SECURITY.md): the chain makes edits, deletions and
reordering of *stored* events DETECTABLE. It does not make them impossible,
and an attacker who can rewrite the whole database AND recompute all hashes
can forge a consistent chain. Signed checkpoints kept outside the database
(create_checkpoint / verify_chain(checkpoint=...)) close most of that gap.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.audit_event import AuditEvent
from app.utils.hashing import (
    GENESIS_HASH,
    canonical_json,
    compute_audit_hash,
    hmac_sha256_hex,
)

log = logging.getLogger(__name__)

# Serialises "read last hash -> insert next event -> commit" inside this
# process. Across processes the UNIQUE(seq) primary key makes a concurrent
# fork fail loudly instead of silently branching the chain.
_chain_lock = threading.RLock()

_SENSITIVE_KEY = re.compile(r"pass(word)?|secret|token|authorization|api[_-]?key|cookie|credential|hash$",
                            re.IGNORECASE)
_MAX_STR = 500


class AuditWriteError(RuntimeError):
    """The audit event (and therefore the business change) was NOT persisted."""


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sanitize_metadata(value: Any, _key: str = "") -> Any:
    """Redact anything that looks like a credential and cap string sizes."""
    if _key and _SENSITIVE_KEY.search(_key) and _key not in {"image_sha256", "model_sha256"}:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): sanitize_metadata(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(v) for v in value][:200]
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "...[truncated]"
    if isinstance(value, float):
        return round(value, 6)
    if value is None or isinstance(value, (int, bool)):
        return value
    return str(value)[:_MAX_STR]


def _hash_fields(ev: AuditEvent) -> dict[str, Any]:
    """Exactly the fields covered by the hash, in stored form."""
    return {
        "seq": ev.seq,
        "event_id": ev.event_id,
        "timestamp": ev.timestamp,
        "actor_id": ev.actor_id,
        "actor_role": ev.actor_role,
        "action": ev.action,
        "target_type": ev.target_type,
        "target_id": ev.target_id,
        "metadata_json": ev.metadata_json,
    }


def append_event(
    db: Session,
    action: str,
    *,
    actor=None,
    target_type: str | None = None,
    target_id: str | None = None,
    metadata: dict | None = None,
) -> AuditEvent:
    """Add the next chained event to the CURRENT transaction (no commit).

    Call it inside `audited_transaction(db)` so the event and the business
    change it describes are committed - or rolled back - together.
    """
    with _chain_lock:
        last = db.execute(select(AuditEvent).order_by(AuditEvent.seq.desc()).limit(1)).scalar_one_or_none()
        prev_hash = last.hash if last else GENESIS_HASH
        role = getattr(actor, "role", None)
        ev = AuditEvent(
            seq=(last.seq + 1) if last else 1,
            event_id=str(uuid.uuid4()),
            timestamp=utc_iso_now(),
            actor_id=getattr(actor, "id", None),
            actor_role=role.value if hasattr(role, "value") else role,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata_json=canonical_json(sanitize_metadata(metadata or {})).decode("utf-8"),
            prev_hash=prev_hash,
            hash="",
        )
        ev.hash = compute_audit_hash(_hash_fields(ev), prev_hash)
        db.add(ev)
        try:
            db.flush()
        except SQLAlchemyError as exc:
            raise AuditWriteError("Audit event could not be written.") from exc
        return ev


@contextmanager
def audited_transaction(db: Session) -> Iterator[None]:
    """Commit the business change and its audit event(s) atomically.

    If anything fails - including the audit write or the final COMMIT - the
    whole transaction is rolled back and AuditWriteError is raised, so the
    caller can never report an unlogged change as recorded.
    """
    with _chain_lock:
        try:
            yield
            db.commit()
        except AuditWriteError:
            db.rollback()
            raise
        except SQLAlchemyError as exc:
            db.rollback()
            log.exception("Transaction with audit event failed")
            raise AuditWriteError("Database transaction failed; change was not recorded.") from exc
        except Exception:
            db.rollback()
            raise


def record_standalone(db: Session, action: str, **kwargs) -> AuditEvent:
    """Write a single event in its own transaction (e.g. a failed login)."""
    with audited_transaction(db):
        return append_event(db, action, **kwargs)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
@dataclass
class VerificationResult:
    valid: bool
    total_events: int
    verified_events: int
    last_seq: int | None
    last_hash: str | None
    first_failure_seq: int | None
    first_failure_event_id: str | None
    failure_reason: str | None
    checkpoint_checked: bool
    checked_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def verify_chain(db: Session, checkpoint: dict | None = None, checkpoint_key: str = "") -> VerificationResult:
    """Walk the whole chain from the genesis event and stop at the first problem.

    Detects: edited fields (hash mismatch), deleted or inserted events
    (sequence gap / broken prev_hash link), reordered events (seq is part of
    the hash), and - with a checkpoint - truncation or wholesale rewriting of
    events up to the checkpoint.
    """
    expected_seq, prev_hash, count = 1, GENESIS_HASH, 0
    last: AuditEvent | None = None
    failure: tuple[int | None, str | None, str] | None = None
    hashes_by_seq: dict[int, str] = {}

    stmt = select(AuditEvent).order_by(AuditEvent.seq).execution_options(yield_per=1000)
    for ev in db.execute(stmt).scalars():
        count += 1
        if ev.seq != expected_seq:
            failure = (ev.seq, ev.event_id,
                       f"Sequence gap: expected event #{expected_seq} but found #{ev.seq} "
                       "(events deleted, inserted or reordered).")
            break
        if ev.prev_hash != prev_hash:
            failure = (ev.seq, ev.event_id,
                       "Broken link: prev_hash does not match the hash of the preceding event.")
            break
        if compute_audit_hash(_hash_fields(ev), ev.prev_hash) != ev.hash:
            failure = (ev.seq, ev.event_id, "Content altered: recomputed SHA-256 does not match "
                                            "the stored hash.")
            break
        if checkpoint:
            hashes_by_seq[ev.seq] = ev.hash
        prev_hash, expected_seq, last = ev.hash, expected_seq + 1, ev

    total = db.query(AuditEvent).count()
    verified = (count - 1) if failure else count

    checkpoint_checked = False
    if failure is None and checkpoint:
        checkpoint_checked = True
        problem = _check_checkpoint(checkpoint, checkpoint_key, hashes_by_seq)
        if problem:
            failure = (checkpoint.get("seq"), None, problem)

    return VerificationResult(
        valid=failure is None,
        total_events=total,
        verified_events=verified,
        last_seq=last.seq if last else None,
        last_hash=last.hash if last else None,
        first_failure_seq=failure[0] if failure else None,
        first_failure_event_id=failure[1] if failure else None,
        failure_reason=failure[2] if failure else None,
        checkpoint_checked=checkpoint_checked,
        checked_at=utc_iso_now(),
    )


# --------------------------------------------------------------------------
# Checkpoints (optional stronger protection)
# --------------------------------------------------------------------------
def _checkpoint_message(seq: int, hash_: str, created_at: str) -> bytes:
    return canonical_json({"seq": seq, "hash": hash_, "created_at": created_at, "v": 1})


def create_checkpoint(db: Session, key: str) -> dict:
    """Sign the current chain head. Store the result OUTSIDE this database."""
    if not key:
        raise ValueError("QGENAI_AUDIT_CHECKPOINT_KEY is not configured.")
    last = db.execute(select(AuditEvent).order_by(AuditEvent.seq.desc()).limit(1)).scalar_one_or_none()
    if last is None:
        raise ValueError("The audit log is empty.")
    created = utc_iso_now()
    return {"seq": last.seq, "hash": last.hash, "created_at": created,
            "signature": hmac_sha256_hex(key, _checkpoint_message(last.seq, last.hash, created))}


def _check_checkpoint(cp: dict, key: str, hashes_by_seq: dict[int, str]) -> str | None:
    try:
        seq, hash_, created, sig = int(cp["seq"]), str(cp["hash"]), str(cp["created_at"]), str(cp["signature"])
    except (KeyError, TypeError, ValueError):
        return "Checkpoint is malformed."
    if not key:
        return "Cannot check checkpoint signature: QGENAI_AUDIT_CHECKPOINT_KEY is not configured."
    import hmac as _hmac
    if not _hmac.compare_digest(sig, hmac_sha256_hex(key, _checkpoint_message(seq, hash_, created))):
        return "Checkpoint signature is invalid (checkpoint forged or key changed)."
    if seq not in hashes_by_seq:
        return f"Event #{seq} from the checkpoint no longer exists (log truncated)."
    if hashes_by_seq[seq] != hash_:
        return f"Event #{seq} hash differs from the signed checkpoint (chain was rewritten)."
    return None
