"""Versioned quality policy: read the active version, create new versions."""
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.quality_policy import QualityPolicy
from app.models.user import User
from app.services.audit_service import append_event, audited_transaction

# Initial policy (version 1). Admins change it via PUT /quality/policy.
DEFAULT_POLICY = {
    "default_confidence_threshold": 0.50,
    "operator_min_threshold": 0.40,
    "operator_max_threshold": 0.70,
    "fail_confidence": 0.50,
    "fail_classes": [],          # [] = every non-ignored class is disqualifying
    "ignored_classes": [],
    "max_allowed_defects": 0,
    "require_approved_model": False,
}

POLICY_FIELDS = list(DEFAULT_POLICY)


def get_active_policy(db: Session) -> QualityPolicy:
    policy = db.execute(
        select(QualityPolicy).order_by(QualityPolicy.version.desc()).limit(1)
    ).scalar_one_or_none()
    if policy is None:
        raise RuntimeError("No quality policy configured - the database was not initialised.")
    return policy


def policy_to_dict(p: QualityPolicy) -> dict:
    return {
        "version": p.version,
        "created_at": p.created_at,
        "created_by": p.created_by,
        "change_reason": p.change_reason,
        "default_confidence_threshold": p.default_confidence_threshold,
        "operator_min_threshold": p.operator_min_threshold,
        "operator_max_threshold": p.operator_max_threshold,
        "fail_confidence": p.fail_confidence,
        "fail_classes": json.loads(p.fail_classes),
        "ignored_classes": json.loads(p.ignored_classes),
        "max_allowed_defects": p.max_allowed_defects,
        "require_approved_model": p.require_approved_model,
    }


def _settings_only(d: dict) -> dict:
    return {k: d[k] for k in POLICY_FIELDS}


def ensure_default_policy(db: Session) -> None:
    if db.execute(select(QualityPolicy.version).limit(1)).first():
        return
    with audited_transaction(db):
        p = QualityPolicy(
            change_reason="Initial default policy",
            created_by=None,
            **{**DEFAULT_POLICY,
               "fail_classes": json.dumps(DEFAULT_POLICY["fail_classes"]),
               "ignored_classes": json.dumps(DEFAULT_POLICY["ignored_classes"])},
        )
        db.add(p)
        db.flush()
        append_event(db, "policy.initialized", target_type="policy", target_id=str(p.version),
                     metadata={"policy": DEFAULT_POLICY})


def create_policy_version(db: Session, admin: User, data: dict) -> QualityPolicy:
    old = get_active_policy(db)
    old_values = _settings_only(policy_to_dict(old))
    new_values = {k: data[k] for k in POLICY_FIELDS}
    with audited_transaction(db):
        p = QualityPolicy(
            created_by=admin.id,
            change_reason=data["change_reason"],
            **{**new_values,
               "fail_classes": json.dumps(sorted(set(new_values["fail_classes"]))),
               "ignored_classes": json.dumps(sorted(set(new_values["ignored_classes"])))},
        )
        db.add(p)
        db.flush()
        changed = {k: {"old": old_values[k], "new": new_values[k]}
                   for k in POLICY_FIELDS if old_values[k] != new_values[k]}
        append_event(db, "policy.updated", actor=admin, target_type="policy",
                     target_id=str(p.version),
                     metadata={"previous_version": old.version, "new_version": p.version,
                               "changes": changed, "reason": data["change_reason"]})
    return p
