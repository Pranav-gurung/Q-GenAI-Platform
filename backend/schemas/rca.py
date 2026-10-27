"""GenAI root-cause analysis: evidence (input), LLM output contract, and the API result.

Separation of concerns is encoded in the types:
  * `RCAEvidence` is built ONLY from the deterministic Part 3 trace analysis (no free-form DB dumps).
  * `LLMRCAOutput` is what the model may return: hypotheses, actions, checks, limitations. It has no fields
    for decisions, detections, stage attribution or numeric confidence - the model cannot supply them.
  * `RootCauseAnalysis` combines deterministic `facts` / `observed_evidence` (written by code) with clearly
    separated AI hypotheses. When the LLM is unavailable the facts remain and the AI fields stay empty.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.domain.enums import DecisionOutcome, DefectType, Severity
from backend.domain.fulfillment import FulfillmentStage
from backend.schemas.fulfillment import (AttributionStatus, Classification, CurrentStageAttribution, DefectStageStatus,
                                         EvidenceLevel, EvidenceSource, ExplanationCode, ObservationStatus,
                                         TimelineStatus, TraceKeyType)

CAUSALITY_DISCLAIMER = ("The pipeline analysis identifies the first observed stage where a defect appears based on "
                        "available inspection evidence. It does not prove physical causality.")


class CauseCategory(str, Enum):
    PACKAGING_MATERIAL = "PACKAGING_MATERIAL"
    PACKAGING_PROCESS = "PACKAGING_PROCESS"
    SEALING_PROCESS = "SEALING_PROCESS"
    HANDLING = "HANDLING"
    STORAGE_CONDITION = "STORAGE_CONDITION"
    PICKING_HANDLING = "PICKING_HANDLING"
    TRANSPORT_HANDLING = "TRANSPORT_HANDLING"
    LABELING = "LABELING"
    PRODUCT_CONDITION = "PRODUCT_CONDITION"   # incl. condition on arrival from the supplier
    UNKNOWN = "UNKNOWN"


class RCAStatus(str, Enum):
    COMPLETED = "COMPLETED"            # validated, grounded AI output
    NOT_APPLICABLE = "NOT_APPLICABLE"  # no defect in the item's history: nothing to explain (no LLM call)
    UNAVAILABLE = "UNAVAILABLE"        # LLM disabled / not configured / timeout / rate limit / API or network error
    INVALID_OUTPUT = "INVALID_OUTPUT"  # LLM answered but failed schema or grounding validation (discarded)


class SuggestedPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ------------------------------------------------------------------------------------------------ evidence (input)
class EvidenceDefect(BaseModel):
    defect_type: DefectType
    severity: Severity | None = None
    cv_confidence: float | None = Field(None, description="Model confidence from CV; null for operator input.")
    source: EvidenceSource


class EvidenceObservation(BaseModel):
    inspection_id: str
    stage: FulfillmentStage
    status: ObservationStatus
    decision_source: EvidenceSource
    defects: list[EvidenceDefect] = Field(default_factory=list)


class EvidenceTimelineEntry(BaseModel):
    stage: FulfillmentStage
    status: TimelineStatus
    defects: list[DefectType] = Field(default_factory=list)


class EvidenceAttribution(BaseModel):
    """Part 3 result for one defect type, copied verbatim, plus the stage window the evidence leaves open."""
    defect_type: DefectType
    classification: Classification
    attribution_status: AttributionStatus
    attributed_stage: FulfillmentStage | None
    evidence_level: EvidenceLevel
    first_observed_stage: FulfillmentStage
    previous_inspected_stage: FulfillmentStage | None
    previous_stage_status: TimelineStatus | None
    last_known_good_stage: FulfillmentStage | None
    missing_stages: list[FulfillmentStage]
    current_stage_status: DefectStageStatus | None
    current_stage_attribution: CurrentStageAttribution
    final_state: DefectStageStatus
    requires_review: bool
    explanation_code: ExplanationCode
    flags: list[ExplanationCode]
    candidate_stages: list[FulfillmentStage] = Field(description=(
        "Stages in which the defect could have arisen given the evidence: after the last known good stage up to "
        "and including the first observed stage. Hypotheses may only point at these stages."))


class OperatorNote(BaseModel):
    """Free text written by an operator. UNTRUSTED: passed to the model as data, never as instructions."""
    inspection_id: str
    stage: FulfillmentStage
    note: Annotated[str, StringConstraints(max_length=300)]


class RCAEvidence(BaseModel):
    inspection_id: str
    store_id: str
    sku_id: str
    item_id: str | None = None
    order_id: str | None = None
    trace_key_type: TraceKeyType
    current_stage: FulfillmentStage
    current_decision: DecisionOutcome | None = None
    current_decision_source: EvidenceSource | None = None
    current_defects: list[EvidenceDefect] = Field(default_factory=list)
    timeline: list[EvidenceTimelineEntry]
    observations: list[EvidenceObservation]
    defect_analyses: list[EvidenceAttribution]
    trace_status: AttributionStatus
    trace_requires_review: bool
    trace_flags: list[ExplanationCode] = Field(default_factory=list)
    untrusted_operator_notes: list[OperatorNote] = Field(default_factory=list)
    causality_disclaimer: str = CAUSALITY_DISCLAIMER


# ------------------------------------------------------------------------------------------------ LLM output contract
Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=400)]


class LLMCauseHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: CauseCategory
    hypothesis: Short
    rationale: Short
    related_stage: FulfillmentStage | None = None


class LLMCorrectiveAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Short
    addresses_category: CauseCategory
    responsible_stage: FulfillmentStage | None = None
    priority: SuggestedPriority


class LLMDefectFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defect_type: DefectType
    likely_causes: list[LLMCauseHypothesis] = Field(max_length=5)
    corrective_actions: list[LLMCorrectiveAction] = Field(min_length=1, max_length=6)
    verification_steps: list[Short] = Field(min_length=1, max_length=6)
    limitations: list[Short] = Field(default_factory=list, max_length=5)


class LLMRCAOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1200)]
    findings: list[LLMDefectFinding] = Field(min_length=1, max_length=10)


# ------------------------------------------------------------------------------------------------ API result
class DefectFacts(BaseModel):
    """Deterministic Part 3 facts for one defect type (never produced by the LLM)."""
    first_observed_stage: FulfillmentStage
    last_known_good_stage: FulfillmentStage | None
    previous_inspected_stage: FulfillmentStage | None
    previous_stage_status: TimelineStatus | None
    missing_stages: list[FulfillmentStage]
    classification: Classification
    evidence_level: EvidenceLevel
    attribution_status: AttributionStatus
    attributed_stage: FulfillmentStage | None
    current_stage_attribution: CurrentStageAttribution
    requires_review: bool
    explanation_code: ExplanationCode
    cv_confidence: float | None = None
    severity: Severity | None = None


class DefectRCA(BaseModel):
    defect_type: DefectType
    facts: DefectFacts
    observed_evidence: list[str] = Field(description="System facts, written deterministically from Part 3.")
    likely_causes: list[LLMCauseHypothesis] = Field(default_factory=list, description="AI hypotheses (not facts).")
    corrective_actions: list[LLMCorrectiveAction] = Field(default_factory=list, description="AI suggestions.")
    verification_steps: list[str] = Field(default_factory=list, description="AI suggestions.")
    limitations: list[str] = Field(description="Deterministic limitations of the evidence.")
    ai_limitations: list[str] = Field(default_factory=list, description="Limitations stated by the model.")


class RCAGeneration(BaseModel):
    ai_generated: bool
    provider: str | None = None
    model: str | None = None
    prompt_version: str
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_code: str | None = None
    error_message: str | None = None


class RootCauseAnalysis(BaseModel):
    status: RCAStatus
    inspection_id: str
    current_stage: FulfillmentStage
    requires_review: bool
    summary: str | None = Field(None, description="AI-generated summary (null unless COMPLETED).")
    defects: list[DefectRCA] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    generation: RCAGeneration
    disclaimer: str = ("AI-generated hypotheses grounded in the inspection evidence shown. "
                       + CAUSALITY_DISCLAIMER)
    rag: RAGResult | None = Field(None, description="SOP grounding of the corrective actions (null if RAG disabled).")


from backend.schemas.rag import RAGResult  # noqa: E402  (rag.py imports RCAGeneration from this module)

RootCauseAnalysis.model_rebuild()
