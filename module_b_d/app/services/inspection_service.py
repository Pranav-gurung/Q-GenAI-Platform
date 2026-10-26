"""Orchestrates one inspection: Section B (vision) + Section D (audit).

Transaction design
------------------
1. `inspection.requested` is committed on its own, so even crashed or rejected
   attempts leave a trace.
2. Validation failures are committed as `inspection.rejected`.
3. The annotated image is written to disk, THEN the Inspection row and its
   audit events (`inspection.completed`, optional `quality.threshold_override`,
   `inspection.result_created`) are committed in ONE transaction.
   - If that transaction fails, it is rolled back, the image file is deleted,
     and AuditWriteError propagates -> the API answers 500 "not recorded".
   - The client therefore never receives an inspection id for a result that
     is not both stored and audit-logged.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path, PurePath

from sqlalchemy.orm import Session

from app.config import Settings
from app.models.inspection import Inspection, InspectionOverride
from app.models.user import User
from app.services.audit_service import append_event, audited_transaction, record_standalone
from app.services.authorization import Permission, can_override_inspection, has_permission
from app.services.inference import run_inference
from app.services.inspection_rules import PolicyRules, evaluate
from app.services.policy_service import get_active_policy
from app.utils.image_utils import ImageValidationError, draw_detections, validate_and_decode

log = logging.getLogger(__name__)


class InspectionPermissionError(PermissionError):
    pass


def annotated_dir(settings: Settings) -> Path:
    d = settings.storage_dir / "annotated"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_threshold(user: User, policy, requested: float | None,
                      override_reason: str | None) -> tuple[float, bool]:
    """Return (threshold, is_override). Raise on invalid / unauthorised values."""
    if requested is None:
        return policy.default_confidence_threshold, False
    if not (0.01 <= requested <= 0.99):
        raise ValueError("Confidence threshold must be between 0.01 and 0.99.")
    if policy.operator_min_threshold <= requested <= policy.operator_max_threshold:
        return requested, False
    # Outside the operator range -> restricted threshold override.
    if not has_permission(user, Permission.THRESHOLD_OVERRIDE):
        raise InspectionPermissionError(
            f"Thresholds outside {policy.operator_min_threshold:.2f}-"
            f"{policy.operator_max_threshold:.2f} require a quality manager.")
    if not override_reason or len(override_reason.strip()) < 10:
        raise ValueError("A reason (min. 10 characters) is required for a threshold override.")
    return requested, True


def run_inspection(db: Session, user: User, data: bytes, filename: str | None,
                   requested_threshold: float | None, override_reason: str | None,
                   detector, settings: Settings) -> tuple[Inspection, list[str]]:
    ext = PurePath(filename or "").suffix.lower()[:10]
    record_standalone(db, "inspection.requested", actor=user, target_type="inspection",
                      metadata={"upload_bytes": len(data), "file_extension": ext})

    def reject(reason: str) -> None:
        record_standalone(db, "inspection.rejected", actor=user, target_type="inspection",
                          metadata={"reason": reason})

    policy = get_active_policy(db)
    try:
        threshold, overridden = resolve_threshold(user, policy, requested_threshold, override_reason)
    except (ValueError, InspectionPermissionError) as exc:
        reject(str(exc))
        raise

    start = time.perf_counter()
    try:
        img = validate_and_decode(data, filename, settings.max_upload_bytes, settings.max_image_pixels)
    except ImageValidationError as exc:
        reject(str(exc))
        raise

    out = run_inference(detector, img.rgb, threshold)
    decision = evaluate(out.defects, PolicyRules.from_db(policy), out.model_info)
    annotated_png = draw_detections(img.rgb, out.defects)
    processing_ms = round((time.perf_counter() - start) * 1000.0, 2)

    inspection_id = str(uuid.uuid4())
    image_name = f"{inspection_id}.png"  # server-generated; never derived from user input
    image_path = annotated_dir(settings) / image_name
    image_path.write_bytes(annotated_png)
    original_path = None
    if settings.store_original_images:
        original_path = settings.storage_dir / "originals" / f"{inspection_id}{ext}"
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_bytes(data)

    info = out.model_info
    event_ids: list[str] = []
    try:
        with audited_transaction(db):
            insp = Inspection(
                id=inspection_id, user_id=user.id, site=user.site,
                image_sha256=img.sha256, image_width=img.width, image_height=img.height,
                annotated_image_file=image_name,
                model_status=info.status, model_name=info.name, model_version=info.version,
                model_sha256=info.sha256,
                confidence_threshold=threshold, threshold_overridden=overridden,
                policy_version=policy.version,
                ai_decision=decision.status, ai_decision_reasons=json.dumps(decision.reasons),
                defects_json=json.dumps(out.defects), defect_count=len(out.defects),
                processing_time_ms=processing_ms, inference_time_ms=out.inference_time_ms,
                final_decision=decision.status,
            )
            db.add(insp)
            db.flush()
            ev = append_event(db, "inspection.completed", actor=user, target_type="inspection",
                              target_id=inspection_id,
                              metadata={"model_status": info.status,
                                        "inference_time_ms": out.inference_time_ms,
                                        "processing_time_ms": processing_ms})
            event_ids.append(ev.event_id)
            if overridden:
                ev = append_event(db, "quality.threshold_override", actor=user,
                                  target_type="inspection", target_id=inspection_id,
                                  metadata={"field": "confidence_threshold",
                                            "old_value": policy.default_confidence_threshold,
                                            "new_value": threshold,
                                            "reason": override_reason.strip()})
                event_ids.append(ev.event_id)
            ev = append_event(db, "inspection.result_created", actor=user, target_type="inspection",
                              target_id=inspection_id,
                              metadata={
                                  "ai_decision": decision.status,
                                  "defects": [{"class_name": d["class_name"],
                                               "confidence": d["confidence"]} for d in out.defects],
                                  "defect_count": len(out.defects),
                                  "confidence_threshold": threshold,
                                  "policy_version": policy.version,
                                  "model_version": info.version,
                                  "model_sha256": info.sha256,
                                  "image_sha256": img.sha256,
                              })
            event_ids.append(ev.event_id)
    except Exception:
        image_path.unlink(missing_ok=True)
        if original_path:
            original_path.unlink(missing_ok=True)
        raise
    return insp, event_ids


def record_override(db: Session, user: User, insp: Inspection, new_decision: str,
                    reason: str) -> InspectionOverride:
    allowed, why = can_override_inspection(user, insp)
    if not allowed:
        raise InspectionPermissionError(why)
    reason = reason.strip()
    if len(reason) < 10:
        raise ValueError("A reason of at least 10 characters is required.")
    old = insp.final_decision
    with audited_transaction(db):
        ov = InspectionOverride(inspection_id=insp.id, actor_id=user.id, actor_role=user.role.value,
                                old_decision=old, new_decision=new_decision, reason=reason)
        db.add(ov)
        # The ORIGINAL AI result (ai_decision, defects_json) is NOT modified.
        insp.final_decision = new_decision
        db.flush()
        append_event(db, "inspection.manual_override", actor=user, target_type="inspection",
                     target_id=insp.id,
                     metadata={"override_id": ov.id, "ai_decision": insp.ai_decision,
                               "old_value": old, "new_value": new_decision, "reason": reason})
    return ov
