"""Fulfillment pipeline intelligence: observation history and per-defect stage attribution.

All values are produced by deterministic rules (backend/fulfillment/analyzer.py). Nothing here is a
probability or a model confidence, and nothing claims physical causality: the analysis identifies the
FIRST OBSERVED stage at which a defect appears, based on the inspections that actually exist.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from backend.domain.enums import DecisionOutcome, DefectType, Severity
from backend.domain.fulfillment import FulfillmentStage


class ObservationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"  # REVIEW decision, or no usable decision (quality rejected, CV unavailable, failed run)


class TimelineStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    NOT_OBSERVED = "NOT_OBSERVED"  # stage not inspected: never assumed good


class EvidenceSource(str, Enum):
    MODEL = "MODEL"            # CV detections + rules decision
    OPERATOR = "OPERATOR"      # operator feedback (overrides the system result)
    NO_DECISION = "NO_DECISION"


class DefectStageStatus(str, Enum):
    NOT_PRESENT = "NOT_PRESENT"        # stage inspected (PASS/FAIL), defect not observed
    FIRST_OBSERVED = "FIRST_OBSERVED"  # earliest inspected stage showing the defect
    PERSISTING = "PERSISTING"          # observed again after its first observation
    RESOLVED = "RESOLVED"              # inspected after first observation, defect no longer observed
    UNCERTAIN = "UNCERTAIN"            # REVIEW stage, or possible re-classification of the same damage


class Classification(str, Enum):
    NEW_DEFECT = "NEW_DEFECT"                        # first seen right after a good, directly preceding inspection
    PRE_EXISTING = "PRE_EXISTING"                    # already present at the first dark-store inspection (receiving)
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # no prior inspection, or uninspected stages in between
    UNCERTAIN = "UNCERTAIN"                          # conflicting / review evidence: a human must look


class AttributionStatus(str, Enum):
    ATTRIBUTED = "ATTRIBUTED"                        # first appearance localised to one stage
    NOT_ATTRIBUTED = "NOT_ATTRIBUTED"                # no defect, or present on arrival (not a dark-store stage)
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class EvidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class CurrentStageAttribution(str, Enum):
    SUPPORTED = "SUPPORTED"          # evidence localises the defect's first appearance to this stage
    NOT_SUPPORTED = "NOT_SUPPORTED"  # defect was already observed at an earlier stage
    UNDETERMINED = "UNDETERMINED"    # this stage shows the defect first, but evidence is insufficient/uncertain
    NOT_APPLICABLE = "NOT_APPLICABLE"  # defect not present at this stage


class ExplanationCode(str, Enum):
    NEW_DEFECT_AFTER_GOOD_STAGE = "NEW_DEFECT_AFTER_GOOD_STAGE"
    NEW_DEFECT_AFTER_DEFECTIVE_STAGE = "NEW_DEFECT_AFTER_DEFECTIVE_STAGE"  # previous stage FAILED for another defect
    DEFECT_PREEXISTING = "DEFECT_PREEXISTING"
    DEFECT_PERSISTING = "DEFECT_PERSISTING"
    DEFECT_RESOLVED = "DEFECT_RESOLVED"
    MISSING_INTERMEDIATE_OBSERVATIONS = "MISSING_INTERMEDIATE_OBSERVATIONS"
    PREVIOUS_STAGE_REVIEW = "PREVIOUS_STAGE_REVIEW"
    AMBIGUOUS_DEFECT_IDENTITY = "AMBIGUOUS_DEFECT_IDENTITY"
    FIRST_OBSERVATION_UNCONFIRMED = "FIRST_OBSERVATION_UNCONFIRMED"
    TIMESTAMP_ORDER_CONFLICT = "TIMESTAMP_ORDER_CONFLICT"
    NO_PRIOR_OBSERVATION = "NO_PRIOR_OBSERVATION"
    NO_DEFECT_OBSERVED = "NO_DEFECT_OBSERVED"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    DUPLICATE_OBSERVATIONS_IGNORED = "DUPLICATE_OBSERVATIONS_IGNORED"
    NO_TRACE_IDENTIFIER = "NO_TRACE_IDENTIFIER"


class TraceKeyType(str, Enum):
    ITEM = "ITEM"    # store + SKU + item_id (unit / handling-unit barcode)
    ORDER = "ORDER"  # store + SKU + order_id (when no item_id)
    NONE = "NONE"    # neither: the inspection cannot be linked to others


class TraceKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    key_type: TraceKeyType
    store_id: str
    sku_id: str
    item_id: str | None = None
    order_id: str | None = None


class ObservedDefect(BaseModel):
    defect_type: DefectType
    severity: Severity | None = None
    confidence: float | None = Field(None, ge=0, le=1, description="Model confidence (null for operator input).")
    source: EvidenceSource


class PipelineObservation(BaseModel):
    """One inspection, normalised for stage analysis (built from persisted results, never re-inferred)."""
    inspection_id: str
    stage: FulfillmentStage
    observed_at: datetime = Field(description="When the inspected image was received (UTC).")
    status: ObservationStatus
    decision: DecisionOutcome | None = None
    decision_source: EvidenceSource
    defects: list[ObservedDefect] = Field(default_factory=list)
    model_version: str | None = None
    image_id: str | None = None
    # Trace fields (checked by the analyzer) and the dedup key. Not part of the API contract and not persisted in
    # snapshots, so they are optional on re-load; build_observation() always sets them.
    store_id: str | None = Field(None, exclude=True)
    sku_id: str | None = Field(None, exclude=True)
    item_id: str | None = Field(None, exclude=True)
    order_id: str | None = Field(None, exclude=True)
    image_sha256: str | None = Field(None, exclude=True)

    @property
    def defect_types(self) -> set[DefectType]:
        return {d.defect_type for d in self.defects}


class TimelineEntry(BaseModel):
    stage: FulfillmentStage
    position: int = Field(description="Canonical 1-based stage order.")
    status: TimelineStatus
    inspection_ids: list[str] = Field(default_factory=list)
    observed_at: datetime | None = None
    defects: list[DefectType] = Field(default_factory=list)


class DefectStageEntry(BaseModel):
    stage: FulfillmentStage
    status: DefectStageStatus


class DefectAnalysis(BaseModel):
    defect_type: DefectType
    classification: Classification
    attribution_status: AttributionStatus
    attributed_stage: FulfillmentStage | None = Field(None, description="Set only when ATTRIBUTED.")
    evidence_level: EvidenceLevel
    evidence_checks: dict[str, bool] = Field(description="The deterministic checks behind evidence_level.")
    first_observed_stage: FulfillmentStage
    first_observed_inspection_id: str
    previous_inspected_stage: FulfillmentStage | None = None
    previous_stage_status: TimelineStatus | None = None
    last_known_good_stage: FulfillmentStage | None = None
    missing_stages: list[FulfillmentStage] = Field(
        default_factory=list, description="Uninspected stages between the previous inspection and the first observation "
                                          "(or before the first observation when there is no previous inspection).")
    stage_statuses: list[DefectStageEntry] = Field(default_factory=list)
    final_state: DefectStageStatus
    current_stage_status: DefectStageStatus | None = None
    current_stage_attribution: CurrentStageAttribution
    requires_review: bool
    explanation_code: ExplanationCode
    flags: list[ExplanationCode] = Field(default_factory=list)


class FulfillmentTraceAnalysis(BaseModel):
    """GET /api/inspections/{id}/pipeline and PipelineAnalysis.trace_analysis."""
    analyzer_version: str
    trace: TraceKey
    current_inspection_id: str
    current_stage: FulfillmentStage
    status: AttributionStatus
    explanation_code: ExplanationCode
    requires_review: bool
    inspected_stages: list[FulfillmentStage]
    missing_stages: list[FulfillmentStage] = Field(
        default_factory=list, description="Uninspected stages between the first and last inspected stage.")
    timeline: list[TimelineEntry]
    defect_analyses: list[DefectAnalysis]
    observations: list[PipelineObservation]
    duplicates_ignored: int = 0
    flags: list[ExplanationCode] = Field(default_factory=list)
    disclaimer: str = ("Identifies the first observed stage at which a defect appears based on available inspection "
                       "evidence. It does not prove physical causality.")
