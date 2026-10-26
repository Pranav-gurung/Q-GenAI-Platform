"""ORM -> API schema mapping (keeps ORM objects out of the API layer)."""
from __future__ import annotations

from backend.db.models import Feedback, Inspection, InspectionImage, InspectionResult
from backend.schemas.feedback import FeedbackRecord
from backend.schemas.image import ImageMetadata
from backend.schemas.inspection import (InspectionDetail, InspectionEvent, InspectionResultOut, PipelineError,
                                        PipelineEventOut)
from backend.schemas.results import (ActionRecommendation, CVResult, ImageQualityResult, PipelineAnalysis,
                                     QualityDecision, RCAResult)

RESULT_SLOTS = {
    "quality": ImageQualityResult, "cv": CVResult, "decision": QualityDecision,
    "pipeline_analysis": PipelineAnalysis, "rca": RCAResult, "action": ActionRecommendation,
}


def image_to_schema(image: InspectionImage) -> ImageMetadata:
    return ImageMetadata.model_validate(image)


def result_to_schema(result: InspectionResult) -> InspectionResultOut:
    payload = {slot: (cls.model_validate(getattr(result, slot)) if getattr(result, slot) else None)
               for slot, cls in RESULT_SLOTS.items()}
    return InspectionResultOut(result_id=result.result_id, updated_at=result.updated_at, **payload)


def inspection_to_event(insp: Inspection) -> InspectionEvent:
    image = insp.current_image
    return InspectionEvent(
        inspection_id=insp.inspection_id, store_id=insp.store_id, sku_id=insp.sku_id, order_id=insp.order_id,
        item_id=insp.item_id, stage=insp.stage, status=insp.status, source=insp.source, operator_id=insp.operator_id,
        image=image_to_schema(image) if image else None,
        result_id=insp.result.result_id if insp.result else None,
        decision=insp.decision,
        model_versions=dict(insp.result.model_versions) if insp.result else {},
        metadata=dict(insp.metadata_ or {}),
        error=PipelineError(code=insp.error_code, message=insp.error_message or "", stage=insp.error_stage)
        if insp.error_code else None,
        created_at=insp.created_at, updated_at=insp.updated_at,
    )


def inspection_to_detail(insp: Inspection) -> InspectionDetail:
    return InspectionDetail(
        **inspection_to_event(insp).model_dump(),
        result=result_to_schema(insp.result) if insp.result else None,
        history=[PipelineEventOut.model_validate(e) for e in insp.events],
    )


def feedback_to_schema(fb: Feedback) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=fb.feedback_id, inspection_id=fb.inspection_id, result_id=fb.result_id,
        operator_id=fb.operator_id, outcome=fb.outcome, defect_types=fb.defect_types or [],
        corrected_detections=fb.corrected_detections or [], suspected_stage=fb.suspected_stage,
        comment=fb.comment, system_outcome=fb.system_outcome, is_correction=fb.is_correction,
        model_versions=dict(fb.model_versions or {}), learning_status=fb.learning_status,
        created_at=fb.created_at,
    )
