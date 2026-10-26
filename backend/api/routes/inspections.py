"""Inspection endpoints. HTTP concerns only; logic lives in backend/services."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Query, UploadFile, status

from backend.api.dependencies import (get_feedback_service, get_inspection_service, get_settings_dep,
                                      require_api_key)
from backend.api.errors import ERROR_RESPONSES
from backend.core.config import Settings
from backend.core.exceptions import InvalidImageError
from backend.domain.fulfillment import FulfillmentStage
from backend.pipeline.states import PipelineState
from backend.schemas.common import ID_PATTERN, Page
from backend.schemas.feedback import FeedbackCreate, FeedbackRecord
from backend.schemas.inspection import InspectionCreate, InspectionDetail, InspectionEvent
from backend.services.feedback_service import FeedbackService
from backend.services.inspection_service import InspectionFilters, InspectionService
from backend.services.mappers import feedback_to_schema, inspection_to_detail, inspection_to_event

router = APIRouter(prefix="/inspections", tags=["inspections"], responses=ERROR_RESPONSES)

InspectionIdPath = Annotated[str, Path(pattern=r"^INS-[0-9A-F]{16}$", examples=["INS-3F9A1C2B7D4E5A60"])]
Service = Annotated[InspectionService, Depends(get_inspection_service)]


@router.post("", status_code=status.HTTP_201_CREATED, response_model=InspectionEvent,
             dependencies=[Depends(require_api_key)], summary="Create an inspection (state CREATED)")
def create_inspection(body: InspectionCreate, service: Service) -> InspectionEvent:
    return inspection_to_event(service.create(body))


@router.get("", response_model=Page[InspectionEvent], summary="List inspections (newest first)")
def list_inspections(
    service: Service,
    store_id: Annotated[str | None, Query(pattern=ID_PATTERN)] = None,
    sku_id: Annotated[str | None, Query(pattern=ID_PATTERN)] = None,
    stage: FulfillmentStage | None = None,
    status_: Annotated[PipelineState | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[InspectionEvent]:
    items, total = service.list(InspectionFilters(store_id, sku_id, stage, status_), limit, offset)
    return Page(items=[inspection_to_event(i) for i in items], total=total, limit=limit, offset=offset)


@router.get("/{inspection_id}", response_model=InspectionDetail,
            summary="Inspection with module results and pipeline history")
def get_inspection(inspection_id: InspectionIdPath, service: Service) -> InspectionDetail:
    return inspection_to_detail(service.get(inspection_id))


@router.post("/{inspection_id}/image", response_model=InspectionDetail, dependencies=[Depends(require_api_key)],
             summary="Upload the inspection image and run the pipeline",
             description="Validates the image (type, size, integrity), stores it, moves the inspection to "
                         "IMAGE_RECEIVED and runs the pipeline synchronously. Allowed from CREATED or "
                         "REQUIRES_REVIEW (retake).")
async def upload_image(inspection_id: InspectionIdPath, service: Service,
                       settings: Annotated[Settings, Depends(get_settings_dep)],
                       file: Annotated[UploadFile, File(description="JPEG / PNG / WebP")]) -> InspectionDetail:
    data = await file.read(settings.max_upload_bytes + 1)  # bounded read
    if len(data) > settings.max_upload_bytes:
        raise InvalidImageError(f"Image exceeds maximum size of {settings.max_upload_mb:g} MB",
                                code="IMAGE_TOO_LARGE", details={"max_bytes": settings.max_upload_bytes})
    insp = service.attach_image_and_run(inspection_id, data, file.filename, file.content_type)
    return inspection_to_detail(insp)


@router.post("/{inspection_id}/feedback", status_code=status.HTTP_201_CREATED, response_model=FeedbackRecord,
             dependencies=[Depends(require_api_key)], summary="Submit operator feedback (active-learning input)")
def submit_feedback(inspection_id: InspectionIdPath, body: FeedbackCreate,
                    feedback: Annotated[FeedbackService, Depends(get_feedback_service)]) -> FeedbackRecord:
    return feedback_to_schema(feedback.submit(inspection_id, body))
