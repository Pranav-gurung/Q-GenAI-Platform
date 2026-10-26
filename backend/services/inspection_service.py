"""Inspection use cases: create, query, attach image + run pipeline.

This is the only place that combines the database, image storage and the pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.core.config import Settings
from backend.core.exceptions import NotFoundError, ValidationFailedError
from backend.core.logging import log_pipeline
from backend.db.models import Inspection, InspectionImage, InspectionResult, PipelineEvent, Sku, Store
from backend.domain.fulfillment import FulfillmentStage
from backend.pipeline.context import ImageInput, InspectionInfo, PipelineContext
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.states import PipelineState
from backend.pipeline.transitions import ensure_transition
from backend.schemas.common import new_id, utcnow
from backend.schemas.inspection import InspectionCreate
from backend.services.image_validation import validate_image
from backend.services.mappers import RESULT_SLOTS, image_to_schema
from backend.services.storage import ImageStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InspectionFilters:
    store_id: str | None = None
    sku_id: str | None = None
    stage: FulfillmentStage | None = None
    status: PipelineState | None = None


class InspectionService:
    def __init__(self, session: Session, storage: ImageStorage, orchestrator: PipelineOrchestrator,
                 settings: Settings) -> None:
        self.session = session
        self.storage = storage
        self.orchestrator = orchestrator
        self.settings = settings

    # ------------------------------------------------------------------ create / read
    def create(self, data: InspectionCreate) -> Inspection:
        if self.session.get(Store, data.store_id) is None:
            raise ValidationFailedError(f"Unknown store '{data.store_id}'", code="UNKNOWN_STORE")
        if self.session.get(Sku, data.sku_id) is None:
            raise ValidationFailedError(f"Unknown SKU '{data.sku_id}'", code="UNKNOWN_SKU")

        now = utcnow()
        insp = Inspection(
            inspection_id=new_id("INS"), store_id=data.store_id, sku_id=data.sku_id, order_id=data.order_id,
            stage=data.stage, status=PipelineState.CREATED, source=data.source.value,
            operator_id=data.operator_id, metadata_=dict(data.metadata), created_at=now, updated_at=now,
        )
        insp.events.append(PipelineEvent(event_type="STATE_TRANSITION", to_state=PipelineState.CREATED,
                                         message="Inspection received", created_at=now))
        self.session.add(insp)
        self.session.commit()
        log_pipeline(insp.inspection_id, PipelineState.CREATED.value, "RECEIVED",
                     message=f"store={insp.store_id} sku={insp.sku_id} stage={insp.stage.value}")
        return insp

    def get(self, inspection_id: str) -> Inspection:
        insp = self.session.scalar(
            select(Inspection).where(Inspection.inspection_id == inspection_id).options(
                selectinload(Inspection.images), selectinload(Inspection.result), selectinload(Inspection.events)))
        if insp is None:
            raise NotFoundError(f"Inspection '{inspection_id}' not found", code="INSPECTION_NOT_FOUND")
        return insp

    def list(self, filters: InspectionFilters, limit: int, offset: int) -> tuple[list[Inspection], int]:
        conditions = [getattr(Inspection, f) == v for f, v in vars(filters).items() if v is not None]
        total = self.session.scalar(select(func.count()).select_from(Inspection).where(*conditions)) or 0
        rows = self.session.scalars(
            select(Inspection).where(*conditions).options(selectinload(Inspection.images),
                                                          selectinload(Inspection.result))
            .order_by(Inspection.created_at.desc(), Inspection.inspection_id).limit(limit).offset(offset)).all()
        return list(rows), total

    # ------------------------------------------------------------------ image + pipeline
    def attach_image_and_run(self, inspection_id: str, data: bytes, filename: str | None,
                             content_type: str | None) -> Inspection:
        insp = self.get(inspection_id)
        ensure_transition(insp.status, PipelineState.IMAGE_RECEIVED)  # 409 if e.g. already COMPLETED
        img = validate_image(data, filename, content_type, allowed_types=self.settings.allowed_image_types,
                             max_bytes=self.settings.max_upload_bytes, max_pixels=self.settings.max_image_pixels)

        image_id = new_id("IMG")
        stored = self.storage.save(f"{inspection_id}/{image_id}{img.extension}", img.data, img.mime_type)
        try:
            now = utcnow()
            image_row = InspectionImage(
                image_id=image_id, inspection_id=inspection_id, file_name=img.file_name, mime_type=img.mime_type,
                size_bytes=img.size_bytes, width=img.width, height=img.height, sha256=img.sha256,
                storage_backend=stored.backend, storage_reference=stored.reference, created_at=now)
            insp.images.append(image_row)
            insp.events.append(PipelineEvent(event_type="STATE_TRANSITION", from_state=insp.status,
                                             to_state=PipelineState.IMAGE_RECEIVED, created_at=now,
                                             message=f"Image {image_id} stored ({img.width}x{img.height} "
                                                     f"{img.image_type})"))
            insp.status = PipelineState.IMAGE_RECEIVED
            insp.decision = None
            insp.error_code = insp.error_message = insp.error_stage = None
            self.session.commit()
        except Exception:
            self.session.rollback()
            self.storage.delete(stored.reference)  # don't leave orphaned files
            raise
        log_pipeline(inspection_id, PipelineState.IMAGE_RECEIVED.value, "COMPLETED", message=f"image={image_id}")
        return self.run_pipeline(insp)

    def run_pipeline(self, insp: Inspection) -> Inspection:
        image_row = insp.current_image
        reference = image_row.storage_reference
        ctx = PipelineContext(
            inspection=InspectionInfo(inspection_id=insp.inspection_id, store_id=insp.store_id, sku_id=insp.sku_id,
                                      stage=insp.stage, order_id=insp.order_id, metadata=dict(insp.metadata_ or {})),
            state=insp.status,
            image=ImageInput(metadata=image_to_schema(image_row), load_bytes=lambda: self.storage.load(reference)),
        )
        self.orchestrator.run(ctx)
        self._persist(insp, ctx)
        return insp

    def _persist(self, insp: Inspection, ctx: PipelineContext) -> None:
        for h in ctx.history:
            insp.events.append(PipelineEvent(
                event_type=h.event_type, from_state=h.from_state, to_state=h.to_state, module=h.module,
                module_status=h.module_status, duration_ms=h.duration_ms, error_code=h.error_code,
                message=h.message, created_at=h.created_at))

        result = insp.result or InspectionResult(result_id=new_id("RES"), inspection_id=insp.inspection_id,
                                                 created_at=utcnow())
        for slot in RESULT_SLOTS:
            value = getattr(ctx, slot)
            setattr(result, slot, value.model_dump(mode="json") if value is not None else None)
        result.model_versions = ctx.model_versions()
        result.cv_model_version = ctx.cv.module_version if ctx.cv is not None and ctx.cv.is_real else None
        result.updated_at = utcnow()
        insp.result = result

        insp.status = ctx.state
        insp.decision = ctx.final_decision
        if ctx.errors:
            err = ctx.errors[-1]
            insp.error_code, insp.error_message, insp.error_stage = err.code, err.message[:500], err.stage
        insp.updated_at = utcnow()
        self.session.commit()
