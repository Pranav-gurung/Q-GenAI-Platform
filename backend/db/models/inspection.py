"""Inspection aggregate: inspection, its image(s), pipeline history and module results."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base, JSONType, UTCDateTime, str_enum
from backend.domain.enums import DecisionOutcome, ModuleStatus
from backend.domain.fulfillment import FulfillmentStage
from backend.pipeline.states import PipelineState
from backend.schemas.common import utcnow


class Inspection(Base):
    __tablename__ = "inspections"

    inspection_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    store_id: Mapped[str] = mapped_column(ForeignKey("stores.store_id"), index=True)
    sku_id: Mapped[str] = mapped_column(ForeignKey("skus.sku_id"), index=True)
    order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    item_id: Mapped[str | None] = mapped_column(String(64), index=True)  # unit / handling-unit id (trace key)
    stage: Mapped[FulfillmentStage] = mapped_column(str_enum(FulfillmentStage), index=True)
    status: Mapped[PipelineState] = mapped_column(str_enum(PipelineState), index=True)
    source: Mapped[str] = mapped_column(String(32))
    operator_id: Mapped[str | None] = mapped_column(String(64))
    decision: Mapped[DecisionOutcome | None] = mapped_column(str_enum(DecisionOutcome))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(500))
    error_stage: Mapped[PipelineState | None] = mapped_column(str_enum(PipelineState))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    images: Mapped[list["InspectionImage"]] = relationship(
        back_populates="inspection", order_by="InspectionImage.created_at", cascade="all, delete-orphan")
    result: Mapped["InspectionResult | None"] = relationship(
        back_populates="inspection", uselist=False, cascade="all, delete-orphan")
    events: Mapped[list["PipelineEvent"]] = relationship(
        back_populates="inspection", order_by="PipelineEvent.id", cascade="all, delete-orphan")

    @property
    def current_image(self) -> "InspectionImage | None":
        return self.images[-1] if self.images else None


class InspectionImage(Base):
    __tablename__ = "inspection_images"

    image_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.inspection_id", ondelete="CASCADE"),
                                               index=True)
    file_name: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(50))
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_backend: Mapped[str] = mapped_column(String(32))
    storage_reference: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    inspection: Mapped[Inspection] = relationship(back_populates="images")


class PipelineEvent(Base):
    """Append-only pipeline history: state transitions, module outcomes, errors."""
    __tablename__ = "pipeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.inspection_id", ondelete="CASCADE"),
                                               index=True)
    event_type: Mapped[str] = mapped_column(String(32))  # STATE_TRANSITION | MODULE_RESULT | ERROR
    from_state: Mapped[PipelineState | None] = mapped_column(str_enum(PipelineState))
    to_state: Mapped[PipelineState | None] = mapped_column(str_enum(PipelineState))
    module: Mapped[str | None] = mapped_column(String(64))
    module_status: Mapped[ModuleStatus | None] = mapped_column(str_enum(ModuleStatus))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error_code: Mapped[str | None] = mapped_column(String(64))
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    inspection: Mapped[Inspection] = relationship(back_populates="events")


class InspectionResult(Base):
    """Latest module outputs for an inspection (one row per inspection).

    Each JSON column holds the serialised ModuleResult (including its status), so a
    NOT_IMPLEMENTED module is stored explicitly rather than as a missing value.
    """
    __tablename__ = "inspection_results"

    result_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.inspection_id", ondelete="CASCADE"),
                                               unique=True)
    quality: Mapped[dict | None] = mapped_column(JSONType)
    cv: Mapped[dict | None] = mapped_column(JSONType)
    decision: Mapped[dict | None] = mapped_column(JSONType)
    pipeline_analysis: Mapped[dict | None] = mapped_column(JSONType)
    rca: Mapped[dict | None] = mapped_column(JSONType)
    action: Mapped[dict | None] = mapped_column(JSONType)
    # Denormalised for querying ("what did model X predict?")
    cv_model_version: Mapped[str | None] = mapped_column(String(100), index=True)
    model_versions: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    inspection: Mapped[Inspection] = relationship(back_populates="result")
