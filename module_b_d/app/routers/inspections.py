"""Section B endpoints: run inspections, list/view them, download reports."""
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.dependencies import get_current_user, require_permission
from app.models.inspection import Inspection
from app.models.user import User
from app.schemas.inspection import InspectionList, InspectionResult, RCAPayload, RCARequestOut
from app.services.authorization import Permission, can_view_inspection, inspection_scope_filter
from app.services.detector import get_detector
from app.services.inspection_service import InspectionPermissionError, annotated_dir, run_inspection
from app.services.report_service import (
    UnsupportedDefectClassError,
    build_module_c_rca_request,
    build_rca_payload,
    build_report,
    inspection_to_result,
)
from app.utils.image_utils import ImageValidationError

router = APIRouter(tags=["Inspection"])


def get_visible_inspection(db: Session, user: User, inspection_id: str) -> Inspection:
    """Load an inspection the user may see. Returns 404 (not 403) otherwise,
    so ids of other users' inspections cannot be probed (IDOR protection)."""
    insp = db.get(Inspection, inspection_id)
    if insp is None or not can_view_inspection(user, insp):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Inspection not found.")
    return insp


@router.post("/predict", response_model=InspectionResult, status_code=status.HTTP_201_CREATED,
             summary="Inspect a product image (JPG/PNG)")
def predict(
    file: UploadFile = File(..., description="Product image (.jpg, .jpeg, .png)"),
    confidence_threshold: float | None = Form(None, description="Optional; defaults to policy value"),
    override_reason: str | None = Form(None, max_length=2000,
                                       description="Required when the threshold is outside the operator range"),
    user: User = Depends(require_permission(Permission.INSPECTION_RUN)),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    detector=Depends(get_detector),
):
    # Sync endpoint: FastAPI runs it in a worker thread, so CPU-heavy inference
    # does not block the event loop. Read at most limit+1 bytes so a huge
    # upload cannot exhaust memory.
    data = file.file.read(settings.max_upload_bytes + 1)
    try:
        insp, event_ids = run_inspection(db, user, data, file.filename, confidence_threshold,
                                         override_reason, detector, settings)
    except ImageValidationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    except InspectionPermissionError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return inspection_to_result(db, insp, event_ids)


@router.get("/inspections", response_model=InspectionList,
            summary="List inspections visible to the current user")
def list_inspections(
    ai_decision: Literal["PASS", "FAIL", "REVIEW"] | None = None,
    final_decision: Literal["PASS", "FAIL", "REVIEW"] | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conds = [inspection_scope_filter(user)]
    if ai_decision:
        conds.append(Inspection.ai_decision == ai_decision)
    if final_decision:
        conds.append(Inspection.final_decision == final_decision)
    total = db.execute(select(func.count()).select_from(Inspection).where(*conds)).scalar_one()
    rows = db.execute(
        select(Inspection, User.username).join(User, User.id == Inspection.user_id)
        .where(*conds).order_by(Inspection.created_at.desc()).limit(limit).offset(offset)
    ).all()
    items = [{
        "inspection_id": i.id, "timestamp": i.created_at, "inspected_by": uname, "site": i.site,
        "ai_decision": i.ai_decision, "final_decision": i.final_decision,
        "defect_count": i.defect_count, "model_version": i.model_version,
        "has_override": bool(i.overrides),
    } for i, uname in rows]
    return {"total": total, "items": items}


@router.get("/inspections/{inspection_id}", response_model=InspectionResult,
            summary="Get one inspection (original AI result + overrides)")
def get_inspection(inspection_id: str, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    return inspection_to_result(db, get_visible_inspection(db, user, inspection_id))


@router.get("/inspections/{inspection_id}/report", summary="Download the JSON inspection report")
def download_report(inspection_id: str, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    insp = get_visible_inspection(db, user, inspection_id)
    report = jsonable_encoder(build_report(db, insp))
    return JSONResponse(
        content=report,
        headers={"Content-Disposition": f'attachment; filename="qgenai-report-{insp.id}.json"'},
    )


@router.get("/inspections/{inspection_id}/annotated-image", summary="Download the annotated PNG",
            response_class=FileResponse)
def annotated_image(inspection_id: str, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    insp = get_visible_inspection(db, user, inspection_id)
    # File name comes from the DB (server-generated "<uuid>.png"), never from the request.
    path = annotated_dir(settings) / (insp.annotated_image_file or "")
    if not insp.annotated_image_file or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Annotated image not available.")
    return FileResponse(path, media_type="image/png",
                        filename=f"qgenai-annotated-{insp.id}.png")


@router.get("/inspections/{inspection_id}/rca-payload", response_model=RCAPayload,
            tags=["Integration (Section C)"],
            summary="Structured, image-free inspection data for root-cause analysis")
def rca_payload(inspection_id: str, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    return build_rca_payload(get_visible_inspection(db, user, inspection_id))


@router.get(
    "/inspections/{inspection_id}/rca-request",
    response_model=RCARequestOut,
    response_model_exclude_none=True,
    tags=["Integration (Section C)"],
    summary="Module C compliant RCARequest payload",
)
def rca_request(
    inspection_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    insp = get_visible_inspection(db, user, inspection_id)
    try:
        return build_module_c_rca_request(
            insp,
            supported_classes=settings.rca_supported_classes_list,
        )
    except UnsupportedDefectClassError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": str(exc),
                "unsupported_classes": exc.unsupported_classes,
                "supported_classes": exc.supported_classes,
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

