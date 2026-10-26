"""Inspection reports and the Section C (root-cause analysis) payload."""
import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.audit_event import AuditEvent
from app.models.inspection import Inspection
from app.models.user import User

DISCLAIMER = (
    "This report records the output of an automated visual inspection model and any "
    "subsequent human review. Model confidence scores are not measures of physical "
    "defect severity. This report is NOT a certificate of conformity with BIS, ISO or "
    "any other standard."
)


def inspection_audit_events(db: Session, inspection_id: str) -> list[AuditEvent]:
    return list(db.execute(
        select(AuditEvent).where(AuditEvent.target_id == inspection_id).order_by(AuditEvent.seq)
    ).scalars())


def inspection_to_result(db: Session, insp: Inspection, event_ids: list[str] | None = None) -> dict:
    username = db.execute(select(User.username).where(User.id == insp.user_id)).scalar_one_or_none()
    defects = json.loads(insp.defects_json)
    if event_ids is None:
        event_ids = [e.event_id for e in inspection_audit_events(db, insp.id)]
    created = insp.created_at
    if created.tzinfo is None:  # SQLite returns naive datetimes; they are stored in UTC
        created = created.replace(tzinfo=timezone.utc)
    return {
        "inspection_id": insp.id,
        "timestamp": created,
        "inspected_by": username or "unknown",
        "site": insp.site,
        "model_status": insp.model_status,
        "model_name": insp.model_name,
        "model_version": insp.model_version,
        "model_sha256": insp.model_sha256,
        "policy_version": insp.policy_version,
        "confidence_threshold": insp.confidence_threshold,
        "threshold_overridden": insp.threshold_overridden,
        "processing_time_ms": insp.processing_time_ms,
        "inference_time_ms": insp.inference_time_ms,
        "image_width": insp.image_width,
        "image_height": insp.image_height,
        "image_sha256": insp.image_sha256,
        "inspection_status": insp.ai_decision,
        "defect_count": insp.defect_count,
        "defects": defects,
        "ai_result": {
            "inspection_status": insp.ai_decision,
            "decision_reasons": json.loads(insp.ai_decision_reasons),
            "defect_count": insp.defect_count,
            "defects": defects,
        },
        "final_decision": insp.final_decision,
        "overrides": [
            {"id": o.id, "actor_id": o.actor_id, "actor_role": o.actor_role,
             "old_decision": o.old_decision, "new_decision": o.new_decision,
             "reason": o.reason, "created_at": o.created_at}
            for o in insp.overrides
        ],
        "annotated_image_available": bool(insp.annotated_image_file),
        "audit_event_ids": event_ids,
    }


def build_report(db: Session, insp: Inspection) -> dict:
    events = inspection_audit_events(db, insp.id)
    result = inspection_to_result(db, insp, [e.event_id for e in events])
    return {
        "report_type": "Q-GenAI Inspection Report",
        "report_schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inspection": result,
        "audit_trail": [
            {"seq": e.seq, "event_id": e.event_id, "timestamp": e.timestamp,
             "action": e.action, "actor_role": e.actor_role, "hash": e.hash}
            for e in events
        ],
        "disclaimer": DISCLAIMER,
    }


def build_rca_payload(insp: Inspection) -> dict:
    """Minimal, image-free data for Section C (root-cause analysis / SOPs)."""
    created = insp.created_at if insp.created_at.tzinfo else insp.created_at.replace(tzinfo=timezone.utc)
    return {
        "inspection_id": insp.id,
        "timestamp": created,
        "site": insp.site,
        "model_version": insp.model_version,
        "model_sha256": insp.model_sha256,
        "policy_version": insp.policy_version,
        "confidence_threshold": insp.confidence_threshold,
        "ai_decision": insp.ai_decision,
        "final_decision": insp.final_decision,
        "defects": json.loads(insp.defects_json),
        "image_width": insp.image_width,
        "image_height": insp.image_height,
    }
