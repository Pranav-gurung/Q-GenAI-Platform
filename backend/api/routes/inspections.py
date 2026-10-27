"""Inspection endpoints. HTTP concerns only; logic lives in backend/services."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Query, Request, UploadFile, status

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
from backend.schemas.fulfillment import FulfillmentTraceAnalysis
from backend.schemas.rca import RootCauseAnalysis
from backend.services.rca_service import generate_inspection_rca
from backend.services.fulfillment_service import analyze_inspection_trace
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
    order_id: Annotated[str | None, Query(pattern=ID_PATTERN)] = None,
    item_id: Annotated[str | None, Query(pattern=ID_PATTERN)] = None,
    stage: FulfillmentStage | None = None,
    status_: Annotated[PipelineState | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[InspectionEvent]:
    filters = InspectionFilters(store_id=store_id, sku_id=sku_id, order_id=order_id, item_id=item_id, stage=stage,
                                status=status_)
    items, total = service.list(filters, limit, offset)
    return Page(items=[inspection_to_event(i) for i in items], total=total, limit=limit, offset=offset)


@router.get("/{inspection_id}", response_model=InspectionDetail,
            summary="Inspection with module results and pipeline history")
def get_inspection(inspection_id: InspectionIdPath, service: Service) -> InspectionDetail:
    return inspection_to_detail(service.get(inspection_id))


@router.get("/{inspection_id}/pipeline", response_model=FulfillmentTraceAnalysis,
            summary="Fulfillment history of this inspection's item/order, with first-defect stage attribution",
            description="Observations of the same store + SKU + item_id (or order_id when there is no item_id), ordered "
                        "by fulfillment stage, analysed deterministically on request from persisted results and operator "
                        "feedback. Identifies the first OBSERVED stage of each defect; it does not prove physical "
                        "causality. Uninspected stages are reported as NOT_OBSERVED, never assumed good.")
def get_inspection_pipeline(inspection_id: InspectionIdPath, service: Service) -> FulfillmentTraceAnalysis:
    return analyze_inspection_trace(service.session, service.get(inspection_id))


@router.get("/{inspection_id}/rca", response_model=RootCauseAnalysis,
            summary="GenAI root-cause analysis, grounded in the fulfillment pipeline evidence",
            description="Rebuilds the deterministic Part 3 analysis from persisted inspections and operator feedback, "
                        "then asks the configured LLM for HYPOTHESES (likely contributing causes, corrective actions, "
                        "verification steps). Facts and observed evidence are produced by code, not the LLM. Never "
                        "fails because of the LLM: when it is disabled, unavailable or returns invalid/ungrounded "
                        "output, status is UNAVAILABLE / INVALID_OUTPUT, the facts are still returned and "
                        "requires_review is true. Each call may make one synchronous LLM request.")
def get_inspection_rca(inspection_id: InspectionIdPath, service: Service, request: Request) -> RootCauseAnalysis:
    return generate_inspection_rca(service.session, service.get(inspection_id), request.app.state.rca_service)


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
             dependencies=[Depends(require_api_key)], summary="Submit operator feedback / review of the AI result",
             description="Operator verdict plus an optional structured review of each AI layer (decision, defects, "
                         "first observed stage: correct?; RCA, actions: useful?). Corrections are stored NEXT TO a "
                         "snapshot of the AI output; the AI result is never modified. Every submission is kept. "
                         "Feedback is input for human-reviewed error analysis; nothing is retrained automatically.")
def submit_feedback(inspection_id: InspectionIdPath, body: FeedbackCreate,
                    feedback: Annotated[FeedbackService, Depends(get_feedback_service)]) -> FeedbackRecord:
    return feedback_to_schema(feedback.submit(inspection_id, body))


@router.get("/{inspection_id}/feedback", response_model=list[FeedbackRecord],
            summary="All operator feedback for an inspection (oldest first)")
def list_feedback(inspection_id: InspectionIdPath,
                  feedback: Annotated[FeedbackService, Depends(get_feedback_service)]) -> list[FeedbackRecord]:
    feedback.inspections.get(inspection_id)  # 404 for unknown inspections
    return [feedback_to_schema(fb) for fb in feedback.list_for(inspection_id)]
