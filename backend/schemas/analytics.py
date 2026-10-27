"""Operations intelligence (Part 7): deterministic aggregation of stored inspections and operator feedback.

Every number is computed by code from persisted data. Nothing here is predicted, forecast or produced by an LLM,
and nothing here changes CV, rules, attribution, RCA, RAG or feedback. Rates use explicit denominators (see
DEFINITIONS); a rate is null when its denominator is zero, never 0 and never NaN.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from backend.domain.enums import DefectType
from backend.domain.fulfillment import FulfillmentStage
from backend.schemas.learning import UsefulnessStats

CAUSALITY_NOTE = "Analytics describe observed inspection patterns and should not be interpreted as causal proof."

DEFINITIONS = {
    "reviewable_inspection": "An inspection that reached COMPLETED / REQUIRES_REVIEW / FAILED in the period. "
                             "In-flight inspections are counted separately and excluded from every rate.",
    "outcome": "Per inspection, the Part 3 observation rule: latest operator verdict, else the automated rules "
               "decision. Inspections with neither are IMAGE_REJECTED (quality gate), PIPELINE_FAILED or NO_DECISION.",
    "decided_inspections": "PASS + FAIL + REVIEW outcomes. Denominator of fail_rate and observed_defect_rate.",
    "fail_rate": "FAIL outcomes / decided inspections (REVIEW stays in the denominator; image-rejected and "
                 "undecided inspections are excluded).",
    "observed_defect_rate": "Decided inspections with >= 1 observed defect / decided inspections. Observed defects "
                            "are the operator's defect types when given, else the CV detections. This is a rate "
                            "among INSPECTED items, not a defect rate of all orders or all stock.",
    "defect_observations": "Number of (inspection, defect type) pairs; a defect seen at two stages counts twice.",
    "unique_defects": "Number of (item trace, defect type) pairs from the Part 3 analysis; a defect seen at two "
                      "stages of the same item counts once, at its first observed stage.",
    "first_observed_stage": "The first stage where the defect appears based on available inspection evidence "
                            "within the selected period (Part 3). Not a proven cause.",
    "hotspot": "Stages ranked by unique defects first observed there (ties: observed defect rate, then flow order).",
    "trend": "Compares the observed defect rate of the first and second half of the period's buckets that have "
             "decided inspections; |change| < 5 percentage points is 'stable'.",
    "feedback_rates": "From the latest operator feedback per inspection (Part 6): corrections / reviews of that "
                      "layer; usefulness = useful / rated.",
}


class Period(BaseModel):
    start: date
    end: date = Field(description="Inclusive (UTC dates).")
    days: int
    granularity: str = Field(description="day | week (ISO weeks, Monday start).")
    defaulted: bool = Field(description="True when no dates were given: up to 30 days ending at the latest "
                                          "inspection, starting at the first inspection in that window.")


class InspectionMetrics(BaseModel):
    total: int = Field(description="Reviewable inspections in the period.")
    in_progress: int = Field(description="Excluded from every rate.")
    decided: int
    passed: int = Field(serialization_alias="pass")
    fail: int
    review: int
    image_rejected: int
    pipeline_failed: int
    no_decision: int
    operator_verdicts: int = Field(description="Decided inspections whose outcome comes from operator feedback.")
    defective: int = Field(description="Decided inspections with >= 1 observed defect.")
    fail_rate: float | None
    observed_defect_rate: float | None


class DefectTypeMetrics(BaseModel):
    defect_type: DefectType
    observations: int
    unique_defects: int
    share_of_unique: float | None = Field(description="unique_defects / all unique defects in the period.")
    most_common_first_stage: FulfillmentStage | None
    severity: dict[str, int] = Field(description="Observations by CV severity; UNSPECIFIED for operator labels.")


class StageMetrics(BaseModel):
    stage: FulfillmentStage
    inspections: int = Field(description="Decided inspections at this stage.")
    defective_inspections: int
    observed_defect_rate: float | None
    defect_observations: int
    first_observed_defects: int = Field(description="Unique defects whose first observed stage is this stage.")


class Hotspot(BaseModel):
    rank: int
    stage: FulfillmentStage
    first_observed_defects: int
    share: float | None
    observed_defect_rate: float | None
    top_defect_type: DefectType | None
    evidence_levels: dict[str, int] = Field(description="Part 3 evidence level of those first observations.")


class MatrixRow(BaseModel):
    defect_type: DefectType
    counts: dict[str, int | None] = Field(description="Per stage: unique defects first observed there; null when the "
                                                      "stage had no decided inspection in the period (not inspected).")
    total: int


class DefectStageMatrix(BaseModel):
    stages: list[FulfillmentStage]
    rows: list[MatrixRow] = Field(description="Only defect types observed in the period.")


class GroupMetrics(BaseModel):
    key: str
    name: str | None = None
    category: str | None = None
    inspections: int = Field(description="Decided inspections.")
    fail: int
    fail_rate: float | None
    defective_inspections: int
    observed_defect_rate: float | None
    unique_defects: int
    top_defect_type: DefectType | None


class TrendBucket(BaseModel):
    start: date
    inspections: int = Field(description="Decided inspections created in the bucket.")
    fail: int
    defective_inspections: int
    defect_observations: int
    observed_defect_rate: float | None


class TrendDirection(BaseModel):
    direction: str = Field(description="increased | decreased | stable | insufficient_data")
    first_half_rate: float | None = None
    second_half_rate: float | None = None
    statement: str


class Trends(BaseModel):
    granularity: str
    buckets: list[TrendBucket]
    direction: TrendDirection


class FeedbackMetrics(BaseModel):
    feedback_records: int = Field(description="Latest feedback per inspection in the period.")
    decision_reviews: int
    decision_overrides: int
    decision_override_rate: float | None
    defect_reviews: int
    defect_corrections: int
    defect_correction_rate: float | None
    stage_reviews: int
    stage_corrections: int
    stage_correction_rate: float | None
    stage_corrections_by_ai_stage: dict[str, int]
    rca: UsefulnessStats
    actions: UsefulnessStats
    sop_grounded_actions: UsefulnessStats


class Investigation(BaseModel):
    stage: FulfillmentStage
    headline: str
    reasons: list[str]
    caveat: str = ("This is the highest observed quality hotspot in the selected period and should be investigated "
                   "first. It is not a root-cause finding.")


class AnalyticsOverview(BaseModel):
    period: Period | None = Field(description="Null when there is no inspection data at all.")
    filters: dict[str, str | None]
    has_data: bool
    message: str | None = None
    demo_seed_inspections: int = Field(0, description="Inspections created by the synthetic demo seed script.")
    inspections: InspectionMetrics
    defects: list[DefectTypeMetrics] = Field(description="Ranked: top observed quality issues first.")
    unique_defects: int
    defect_observations: int
    stages: list[StageMetrics]
    hotspots: list[Hotspot] = Field(description="Observed quality hotspots (stages with >= 1 first observation).")
    matrix: DefectStageMatrix
    stores: list[GroupMetrics]
    skus: list[GroupMetrics]
    categories: list[GroupMetrics]
    trends: Trends
    feedback: FeedbackMetrics
    investigation: Investigation | None
    definitions: dict[str, str] = Field(default_factory=lambda: dict(DEFINITIONS))
    disclaimer: str = CAUSALITY_NOTE
