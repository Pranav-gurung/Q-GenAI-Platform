"""PipelineContext: everything accumulated while one inspection moves through the pipeline.

Modules read their inputs from the context and the orchestrator writes their typed results
back into it. The context owns the current state and only changes it via the state machine,
recording every transition / module outcome / error in `history`.

The context is pure in-memory domain state (no DB, no HTTP), so the pipeline can be run
and tested in isolation; the inspection service persists it afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from backend.domain.enums import DecisionOutcome, ModuleStatus
from backend.domain.fulfillment import FulfillmentStage
from backend.pipeline.states import PipelineState
from backend.pipeline.transitions import ensure_transition
from backend.schemas.common import utcnow
from backend.schemas.image import ImageMetadata
from backend.schemas.results import (ActionRecommendation, CVResult, ImageQualityResult, PipelineAnalysis,
                                     QualityDecision, RCAResult)


@dataclass(frozen=True)
class InspectionInfo:
    """Immutable inspection facts modules may use (no ORM objects leak into modules)."""
    inspection_id: str
    store_id: str
    sku_id: str
    stage: FulfillmentStage
    order_id: str | None = None
    metadata: dict = field(default_factory=dict)
    item_id: str | None = None  # physical unit / handling-unit id that links observations across stages


@dataclass(frozen=True)
class ImageInput:
    """What an image-consuming module receives: metadata plus a lazy byte loader.

    The loader hides the storage backend (local disk today, S3/GCS/Azure later).
    """
    metadata: ImageMetadata
    load_bytes: Callable[[], bytes]


@dataclass(frozen=True)
class HistoryEntry:
    event_type: str  # STATE_TRANSITION | MODULE_RESULT | ERROR
    created_at: datetime
    from_state: PipelineState | None = None
    to_state: PipelineState | None = None
    module: str | None = None
    module_status: ModuleStatus | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class ContextError:
    code: str
    message: str
    stage: PipelineState


@dataclass
class PipelineContext:
    inspection: InspectionInfo
    state: PipelineState
    image: ImageInput | None = None

    quality: ImageQualityResult | None = None
    cv: CVResult | None = None
    decision: QualityDecision | None = None
    pipeline_analysis: PipelineAnalysis | None = None
    rca: RCAResult | None = None
    action: ActionRecommendation | None = None

    errors: list[ContextError] = field(default_factory=list)
    history: list[HistoryEntry] = field(default_factory=list)

    # Prior persisted observations of the same item/order trace (read-only input for PIPELINE_ANALYSIS;
    # loaded by the inspection service, so pipeline modules never touch the database).
    trace_observations: tuple = ()

    @property
    def inspection_id(self) -> str:
        return self.inspection.inspection_id

    def transition_to(self, target: PipelineState, *, message: str | None = None,
                      error_code: str | None = None) -> None:
        ensure_transition(self.state, target)
        self.history.append(HistoryEntry("STATE_TRANSITION", utcnow(), from_state=self.state, to_state=target,
                                         message=message, error_code=error_code))
        self.state = target

    def record_module(self, module: str, status: ModuleStatus, duration_ms: float | None,
                      message: str | None = None, error_code: str | None = None) -> None:
        self.history.append(HistoryEntry("MODULE_RESULT", utcnow(), to_state=self.state, module=module,
                                         module_status=status, duration_ms=duration_ms,
                                         message=message, error_code=error_code))

    def record_error(self, code: str, message: str) -> None:
        self.errors.append(ContextError(code, message, self.state))
        self.history.append(HistoryEntry("ERROR", utcnow(), to_state=self.state, error_code=code, message=message))

    @property
    def final_decision(self) -> DecisionOutcome | None:
        """Only a real (COMPLETED) decision counts."""
        return self.decision.outcome if self.decision and self.decision.is_real else None

    def model_versions(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for slot in ("quality", "cv", "decision", "pipeline_analysis", "rca", "action"):
            result = getattr(self, slot)
            if result is not None and result.is_real and result.module_version:
                out[slot] = f"{result.module}:{result.module_version}"
        return out
