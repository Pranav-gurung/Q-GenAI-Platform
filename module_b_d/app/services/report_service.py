"""Inspection reports and the Section C (root-cause analysis) payload."""
import json
import math
import re
from collections.abc import Collection
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


class UnsupportedDefectClassError(ValueError):
    """Raised when an inspection contains defect classes unsupported by Module C."""

    def __init__(self, message: str, unsupported_classes: list[str], supported_classes: list[str]):
        super().__init__(message)
        self.unsupported_classes = unsupported_classes
        self.supported_classes = supported_classes


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")
STATUS_MAPPING = {
    "PASS": "nominal",
    "FAIL": "defect_detected",
    "REVIEW": "review_required",
}
DEFAULT_RCA_CLASSES = ("scratch", "crack", "dent", "corrosion")


def build_module_c_rca_request(
    insp: Inspection,
    supported_classes: Collection[str] | None = None,
    part_type: str | None = None,
    material: str | None = None,
    production_process: str | None = None,
) -> dict:
    """Convert an Inspection into a payload strictly conforming to Module C's RCARequest contract.

    Validates:
    - inspection_id, image_id (from image_sha256), and model_version match Module C ID regex.
    - All defects have supported classes (raises UnsupportedDefectClassError otherwise).
    - Bounding boxes are converted from {x1, y1, x2, y2} to [x_min, y_min, x_max, y_max] in
      original-image pixels with strict finite numeric, positive width/height, and bounds validation.
    - Status is mapped from final_decision ("PASS"->"nominal", "FAIL"->"defect_detected", "REVIEW"->"review_required").
    - Operational context is safely preserved in inspection_context (max 300 chars) without secrets.
    - Strict avoidance of any extra top-level fields forbidden by Module C.
    """
    if not insp.id or not isinstance(insp.id, str) or not ID_RE.match(insp.id.strip()):
        raise ValueError(f"Inspection id is missing or invalid for Module C: {insp.id!r}")
    inspection_id = insp.id.strip()

    if not insp.image_sha256 or not isinstance(insp.image_sha256, str) or not ID_RE.match(insp.image_sha256.strip()):
        raise ValueError(f"Inspection image_sha256 is missing or invalid for Module C image_id: {insp.image_sha256!r}")
    image_id = insp.image_sha256.strip()

    if not insp.model_version or not isinstance(insp.model_version, str) or not ID_RE.match(insp.model_version.strip()):
        raise ValueError(f"Inspection model_version is missing or invalid for Module C: {insp.model_version!r}")
    model_version = insp.model_version.strip()

    w = insp.image_width
    h = insp.image_height
    if w is not None and h is not None:
        if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
            raise ValueError(f"image_width must be a positive integer, got {w}")
        if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
            raise ValueError(f"image_height must be a positive integer, got {h}")
    elif (w is None) != (h is None):
        raise ValueError("image_width and image_height must be given together")

    raw_defects = json.loads(insp.defects_json) if isinstance(insp.defects_json, str) else (insp.defects_json or [])
    if not isinstance(raw_defects, list):
        raise ValueError(f"Defects must be a list, got {type(raw_defects).__name__}")

    supported_set = {c.strip().lower() for c in supported_classes} if supported_classes else set(DEFAULT_RCA_CLASSES)
    unsupported_detected = set()
    processed_defects = []

    for idx, d in enumerate(raw_defects):
        if not isinstance(d, dict):
            raise ValueError(f"Defect at index {idx} must be an object, got {type(d).__name__}")
        raw_class = d.get("class_name") or d.get("class")
        if not raw_class or not isinstance(raw_class, str) or not raw_class.strip():
            raise ValueError(f"Defect at index {idx} missing valid class name: {d}")
        c_norm = raw_class.strip().lower()
        if c_norm not in supported_set:
            unsupported_detected.add(raw_class.strip())

        conf = d.get("confidence")
        if (
            conf is None
            or isinstance(conf, bool)
            or not isinstance(conf, (int, float))
            or not math.isfinite(conf)
            or not (0.0 <= conf <= 1.0)
        ):
            raise ValueError(
                f"Defect at index {idx} ({c_norm}) has invalid confidence {conf!r}; must be finite float in [0.0, 1.0]"
            )

        bbox_raw = d.get("bbox")
        mask_ref = d.get("mask_ref")
        if bbox_raw is None and mask_ref is None:
            raise ValueError(f"Defect at index {idx} must provide bbox or mask_ref")

        bbox_out = None
        if bbox_raw is not None:
            if isinstance(bbox_raw, dict):
                for k in ("x1", "y1", "x2", "y2"):
                    if k not in bbox_raw:
                        raise ValueError(f"Defect at index {idx} bbox dict missing coordinate '{k}'")
                coords = [bbox_raw["x1"], bbox_raw["y1"], bbox_raw["x2"], bbox_raw["y2"]]
            elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                coords = list(bbox_raw)
            else:
                raise ValueError(f"Defect at index {idx} bbox must be 4 coordinates or dict, got {bbox_raw}")

            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in coords):
                raise ValueError(f"Defect at index {idx} bbox coordinates must be finite numbers, got {coords}")

            x0, y0, x1, y1 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
            if x0 < 0.0 or y0 < 0.0:
                raise ValueError(f"Defect at index {idx} bbox coordinates must be non-negative, got {[x0, y0, x1, y1]}")
            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    f"Defect at index {idx} box must have positive width and height (x2 > x1 and y2 > y1), got {[x0, y0, x1, y1]}"
                )
            if w is not None and h is not None:
                if x1 > w or y1 > h:
                    raise ValueError(f"Defect at index {idx} box {[x0, y0, x1, y1]} exceeds image size {w}x{h}")
            bbox_out = [x0, y0, x1, y1]

        item = {
            "class": c_norm,
            "confidence": float(conf),
        }
        if bbox_out is not None:
            item["bbox"] = bbox_out
        if mask_ref is not None:
            if not isinstance(mask_ref, str) or not mask_ref.strip():
                raise ValueError(f"Defect at index {idx} mask_ref must be a non-empty string")
            item["mask_ref"] = mask_ref.strip()
        processed_defects.append(item)

    if unsupported_detected:
        raise UnsupportedDefectClassError(
            f"Inspection contains unsupported defect classes for RCA: {sorted(unsupported_detected)}. Supported classes: {sorted(supported_set)}",
            unsupported_classes=sorted(unsupported_detected),
            supported_classes=sorted(supported_set),
        )

    decision = insp.final_decision or insp.ai_decision
    status_str = STATUS_MAPPING.get(decision, decision.lower() if decision else None)

    ctx_parts = [
        f"site={insp.site}",
        f"policy_v={insp.policy_version}",
        f"ai_decision={insp.ai_decision}",
        f"final_decision={insp.final_decision}",
    ]
    if insp.model_sha256:
        ctx_parts.append(f"model_sha256={insp.model_sha256[:16]}")
    if insp.threshold_overridden:
        ctx_parts.append("threshold_overridden=true")
    inspection_context = "; ".join(ctx_parts)[:300]

    payload = {
        "inspection_id": inspection_id,
        "image_id": image_id,
        "model_version": model_version,
        "defects": processed_defects,
    }
    if status_str:
        payload["status"] = status_str[:64]
    if w is not None and h is not None:
        payload["image_width"] = w
        payload["image_height"] = h
    if inspection_context:
        payload["inspection_context"] = inspection_context
    if part_type:
        payload["part_type"] = str(part_type)[:300]
    if material:
        payload["material"] = str(material)[:300]
    if production_process:
        payload["production_process"] = str(production_process)[:300]

    return payload
