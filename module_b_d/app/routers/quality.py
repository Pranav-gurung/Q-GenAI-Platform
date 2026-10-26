"""Quality management: inspection policy, manual review/overrides, scoped audit view."""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import require_any_permission, require_permission
from app.models.inspection import Inspection
from app.models.quality_policy import QualityPolicy
from app.models.user import User
from app.routers.admin import list_audit_events
from app.routers.inspections import get_visible_inspection
from app.schemas.audit import AuditEventList
from app.schemas.inspection import InspectionResult, OverrideRequest, PolicyOut, PolicyUpdate
from app.services.authorization import Permission
from app.services.inspection_service import InspectionPermissionError, record_override
from app.services.policy_service import create_policy_version, get_active_policy, policy_to_dict
from app.services.report_service import inspection_to_result

router = APIRouter(tags=["Quality management"])


@router.get("/quality/policy", response_model=PolicyOut, summary="Active inspection policy")
def get_policy(user: User = Depends(require_permission(Permission.POLICY_VIEW)),
               db: Session = Depends(get_db)):
    return policy_to_dict(get_active_policy(db))


@router.get("/quality/policy/history", response_model=list[PolicyOut],
            summary="All policy versions (newest first)")
def policy_history(user: User = Depends(require_permission(Permission.POLICY_VIEW)),
                   db: Session = Depends(get_db)):
    rows = db.execute(select(QualityPolicy).order_by(QualityPolicy.version.desc())).scalars()
    return [policy_to_dict(p) for p in rows]


@router.put("/quality/policy", response_model=PolicyOut,
            summary="Create a new policy version (administrator only)")
def update_policy(body: PolicyUpdate,
                  admin: User = Depends(require_permission(Permission.POLICY_EDIT)),
                  db: Session = Depends(get_db)):
    return policy_to_dict(create_policy_version(db, admin, body.model_dump()))


@router.post("/inspections/{inspection_id}/override", response_model=InspectionResult,
             summary="Record a manual review decision (quality manager)")
def override_inspection(inspection_id: str, body: OverrideRequest,
                        user: User = Depends(require_permission(Permission.INSPECTION_OVERRIDE)),
                        db: Session = Depends(get_db)):
    insp = get_visible_inspection(db, user, inspection_id)
    try:
        record_override(db, user, insp, body.new_decision, body.reason)
    except InspectionPermissionError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    db.refresh(insp)
    return inspection_to_result(db, insp)


@router.get("/quality/audit-logs", response_model=AuditEventList,
            summary="Audit events within the caller's scope (managers: own site)")
def scoped_audit_logs(
    action: str | None = Query(None, max_length=64),
    target_id: str | None = Query(None, max_length=64),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_any_permission(Permission.AUDIT_VIEW_SITE, Permission.AUDIT_VIEW_ALL)),
    db: Session = Depends(get_db),
):
    return list_audit_events(db, user, action, target_id, limit, offset)


@router.get("/quality/review-queue", response_model=list[str],
            summary="IDs of FAIL/REVIEW inspections in scope that have no human review yet")
def review_queue(user: User = Depends(require_permission(Permission.INSPECTION_OVERRIDE)),
                 db: Session = Depends(get_db)):
    rows = db.execute(
        select(Inspection).where(Inspection.site == user.site,
                                 Inspection.ai_decision.in_(["FAIL", "REVIEW"]))
        .order_by(Inspection.created_at.desc()).limit(500)
    ).scalars()
    return [i.id for i in rows if not i.overrides]
