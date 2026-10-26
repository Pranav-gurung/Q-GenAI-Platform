"""SHA-256 helpers and the canonical serialisation used by the audit chain."""
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

AUDIT_HASH_DOMAIN = b"qgenai-audit-v1\n"
GENESIS_HASH = "0" * 64  # prev_hash of the very first audit event


def canonical_json(obj: Any) -> bytes:
    """Deterministic, unambiguous byte representation of a JSON-able object.

    - keys sorted, no insignificant whitespace
    - UTF-8, NaN/Infinity rejected
    Two equal objects always produce identical bytes.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_audit_hash(event_fields: dict[str, Any], prev_hash: str) -> str:
    """current_hash = SHA256(domain || canonical_json(event_fields + prev_hash)).

    prev_hash is placed INSIDE the canonical JSON object (key "prev_hash"),
    so there is exactly one way to encode a given (event, prev_hash) pair.
    """
    payload = dict(event_fields)
    payload["prev_hash"] = prev_hash
    return sha256_hex(AUDIT_HASH_DOMAIN + canonical_json(payload))


def hmac_sha256_hex(key: str, message: bytes) -> str:
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()
