"""InspectionEvent: the central domain object, plus API request/response models."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from backend.domain.enums import DecisionOutcome, ModuleStatus
from backend.domain.fulfillment import FulfillmentStage
from backend.pipeline.states import PipelineState
from backend.schemas.common import ExternalId, Metadata, StrictModel
from backend.schemas.image import ImageMetadata
from backend.schemas.results import (ActionRecommendation, CVResult, ImageQualityResult, PipelineAnalysis,
                                     QualityDecision, RCAResult)

INSPECTION_SCHEMA_VERSION = "1.0"


class InspectionSource(str, Enum):
    OPERATOR_APP = "OPERATOR_APP"        # handheld / web UI
    STATION_CAMERA = "STATION_CAMERA"    # fixed camera at a station (future)
    API = "API"                          # system-to-system integration


class InspectionCreate(StrictModel):
    """Request body for POST /api/inspections."""
    model_config = StrictModel.model_config | ConfigDict(json_schema_extra={"example": {
        "store_id": "DS-BLR-001", "sku_id": "SKU-MILK-500ML", "stage": "PACKING",
        "order_id": "ORD-100245", "operator_id": "OP-17", "source": "OPERATOR_APP",
        "metadata": {"station": "PACK-03"}}})

    store_id: ExternalId
    sku_id: ExternalId
    stage: FulfillmentStage
    order_id: ExternalId | None = Field(None, description="Customer order, when the inspection is tied to one "
                                                          "(typically PICKING onwards).")
    operator_id: ExternalId | None = None
    source: InspectionSource = InspectionSource.OPERATOR_APP
    metadata: Metadata = Field(default_factory=dict)


class PipelineError(BaseModel):
    code: str
    message: str
    stage: PipelineState | None = None


class PipelineEventOut(BaseModel):
    """One entry in an inspection's pipeline history."""
    model_config = ConfigDict(from_attributes=True)

    event_type: str = Field(examples=["STATE_TRANSITION", "MODULE_RESULT", "ERROR"])
    from_state: PipelineState | None = None
    to_state: PipelineState | None = None
    module: str | None = None
    module_status: ModuleStatus | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    message: str | None = None
    created_at: datetime


class InspectionResultOut(BaseModel):
    """Everything the pipeline modules produced. A missing/NOT_IMPLEMENTED entry means no result."""
    result_id: str
    quality: ImageQualityResult | None = None
    cv: CVResult | None = None
    decision: QualityDecision | None = None
    pipeline_analysis: PipelineAnalysis | None = None
    rca: RCAResult | None = None
    action: ActionRecommendation | None = None
    updated_at: datetime


class InspectionEvent(BaseModel):
    """The inspection as it flows through the platform (API read model)."""
    schema_version: str = INSPECTION_SCHEMA_VERSION
    inspection_id: str = Field(examples=["INS-3F9A1C2B7D4E5A60"])
    store_id: str
    sku_id: str
    order_id: str | None = None
    stage: FulfillmentStage
    status: PipelineState
    source: InspectionSource
    operator_id: str | None = None
    image: ImageMetadata | None = None
    result_id: str | None = None
    decision: DecisionOutcome | None = Field(None, description="Final outcome, set only by a real decision "
                                                                "(model, rules or operator).")
    model_versions: dict[str, str] = Field(default_factory=dict,
                                           description="module -> version for modules that produced real results.")
    metadata: dict = Field(default_factory=dict)
    error: PipelineError | None = None
    created_at: datetime
    updated_at: datetime


class InspectionDetail(InspectionEvent):
    """GET /api/inspections/{id}: event + module results + pipeline history."""
    result: InspectionResultOut | None = None
    history: list[PipelineEventOut] = Field(default_factory=list)
