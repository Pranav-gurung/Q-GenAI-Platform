"""Deterministic fulfillment-stage analysis over the inspection history of one item / order.

Input: PipelineObservations of ONE trace (same store + SKU + item_id, or + order_id). Output: when each
defect type was FIRST OBSERVED, whether that is attributable to a stage, and how strong the evidence is.
It identifies the first observed stage at which a defect appears based on available inspection evidence.
It does not prove physical causality.

Algorithm (N observations, S <= 6 stages, D defect types):  O(N log N + D * S)

 1. Validate every observation belongs to the trace; drop duplicates (same inspection id, or same image
    content = same photo uploaded twice), keeping the earliest. Duplicates are counted, never double-counted.
 2. Sort by (canonical stage order, observed_at, inspection_id) and merge observations of the same stage:
      stage status  = FAIL if any FAIL, else REVIEW if any REVIEW, else PASS     (REVIEW never becomes PASS)
      stage defects = union of defect types;  "confirmed" = seen in a FAIL or operator-provided observation
 3. Timestamp check: every observation of an earlier stage must not be later than any observation of a
    later stage. A violation sets TIMESTAMP_ORDER_CONFLICT (all attributions become UNCERTAIN).
 4. Identity check (defect identity = defect type): if, between two consecutive inspected stages, one defect
    type disappears while another appears, the change may be a re-classification of the same damage ->
    AMBIGUOUS_DEFECT_IDENTITY for both types at that stage.
 5. Per defect type: first stage observed, previous inspected stage, last known good (PASS) stage, uninspected
    stages in between, then classification (first matching rule wins):
      timestamp conflict                          -> UNCERTAIN              TIMESTAMP_ORDER_CONFLICT
      first observed at RECEIVING                 -> PRE_EXISTING           DEFECT_PREEXISTING
      no previous inspected stage                 -> INSUFFICIENT_EVIDENCE  NO_PRIOR_OBSERVATION
      previous inspected stage was REVIEW         -> UNCERTAIN              PREVIOUS_STAGE_REVIEW
      ambiguous identity at the first stage       -> UNCERTAIN              AMBIGUOUS_DEFECT_IDENTITY
      uninspected stages in between               -> INSUFFICIENT_EVIDENCE  MISSING_INTERMEDIATE_OBSERVATIONS
      otherwise                                   -> NEW_DEFECT             NEW_DEFECT_AFTER_GOOD_STAGE
                                                                            (or ..._AFTER_DEFECTIVE_STAGE)
 6. Evidence level (NOT a probability, NOT model confidence):
      NEW_DEFECT    HIGH   previous stage explicitly PASSED and first observation confirmed (FAIL/operator)
                    MEDIUM exactly one of those two is missing
                    LOW    both missing
      PRE_EXISTING  HIGH if the receiving observation is confirmed, else MEDIUM
      UNCERTAIN     LOW
      INSUFFICIENT_EVIDENCE  INSUFFICIENT
    The individual checks are returned in `evidence_checks`.
 7. Attribution: NEW_DEFECT + HIGH/MEDIUM -> ATTRIBUTED (to the first observed stage); NEW_DEFECT + LOW or
    UNCERTAIN -> REQUIRES_REVIEW; INSUFFICIENT_EVIDENCE -> INSUFFICIENT_EVIDENCE; PRE_EXISTING -> NOT_ATTRIBUTED
    (present on arrival, i.e. before any dark-store stage).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from backend.domain.enums import DefectType
from backend.domain.fulfillment import FULFILLMENT_FLOW, STAGE_ORDER, FulfillmentStage, stages_between
from backend.schemas.fulfillment import (AttributionStatus, Classification, CurrentStageAttribution,
                                         DefectAnalysis, DefectStageEntry, DefectStageStatus, EvidenceLevel,
                                         EvidenceSource, ExplanationCode, FulfillmentTraceAnalysis,
                                         ObservationStatus, PipelineObservation, TimelineEntry, TimelineStatus,
                                         TraceKey, TraceKeyType)

ANALYZER_VERSION = "1"
E = ExplanationCode


class TraceMismatchError(ValueError):
    """An observation does not belong to the trace being analysed (would mix items / orders)."""


@dataclass
class StageRecord:
    stage: FulfillmentStage
    observations: list[PipelineObservation] = field(default_factory=list)
    defects: set[DefectType] = field(default_factory=set)
    confirmed: set[DefectType] = field(default_factory=set)
    ambiguous: set[DefectType] = field(default_factory=set)  # types involved in an identity swap at this stage

    @property
    def status(self) -> ObservationStatus:
        statuses = {o.status for o in self.observations}
        if ObservationStatus.FAIL in statuses:
            return ObservationStatus.FAIL
        if ObservationStatus.REVIEW in statuses:
            return ObservationStatus.REVIEW
        return ObservationStatus.PASS

    @property
    def first_at(self) -> datetime:
        return self.observations[0].observed_at

    @property
    def last_at(self) -> datetime:
        return max(o.observed_at for o in self.observations)


def _belongs(obs: PipelineObservation, trace: TraceKey) -> bool:
    if obs.store_id != trace.store_id or obs.sku_id != trace.sku_id:
        return False
    if trace.key_type is TraceKeyType.ITEM:
        return obs.item_id == trace.item_id
    if trace.key_type is TraceKeyType.ORDER:
        return obs.order_id == trace.order_id
    return True


def _dedupe(observations: Iterable[PipelineObservation]) -> tuple[list[PipelineObservation], int]:
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    kept, dropped = [], 0
    for o in sorted(observations, key=lambda o: (o.observed_at, o.inspection_id)):  # earliest wins
        if o.inspection_id in seen_ids or (o.image_sha256 and o.image_sha256 in seen_images):
            dropped += 1
            continue
        seen_ids.add(o.inspection_id)
        if o.image_sha256:
            seen_images.add(o.image_sha256)
        kept.append(o)
    kept.sort(key=lambda o: (STAGE_ORDER[o.stage], o.observed_at, o.inspection_id))
    return kept, dropped


def _stage_records(observations: list[PipelineObservation]) -> list[StageRecord]:
    records: dict[FulfillmentStage, StageRecord] = {}
    for o in observations:
        rec = records.setdefault(o.stage, StageRecord(o.stage))
        rec.observations.append(o)
        types = o.defect_types
        rec.defects |= types
        if o.status is ObservationStatus.FAIL or o.decision_source is EvidenceSource.OPERATOR:
            rec.confirmed |= types
    ordered = sorted(records.values(), key=lambda r: STAGE_ORDER[r.stage])
    for prev, cur in zip(ordered, ordered[1:]):
        appeared, disappeared = cur.defects - prev.defects, prev.defects - cur.defects
        if appeared and disappeared:
            cur.ambiguous |= appeared | disappeared
    return ordered


def _timestamp_conflicts(records: list[StageRecord]) -> list[tuple[FulfillmentStage, FulfillmentStage]]:
    conflicts = []
    latest_so_far: tuple[datetime, FulfillmentStage] | None = None
    for rec in records:
        if latest_so_far and rec.first_at < latest_so_far[0]:
            conflicts.append((latest_so_far[1], rec.stage))
        if latest_so_far is None or rec.last_at > latest_so_far[0]:
            latest_so_far = (rec.last_at, rec.stage)
    return conflicts


def _timeline_status(status: ObservationStatus) -> TimelineStatus:
    return TimelineStatus(status.value)


def _analyse_defect(defect: DefectType, records: list[StageRecord], conflicts: bool,
                    current_stage: FulfillmentStage) -> DefectAnalysis:
    idx = next(i for i, r in enumerate(records) if defect in r.defects)
    first = records[idx]
    prev = records[idx - 1] if idx > 0 else None
    missing = (list(stages_between(prev.stage, first.stage)) if prev
               else list(FULFILLMENT_FLOW[: STAGE_ORDER[first.stage] - 1]))
    last_good = next((r.stage for r in reversed(records[:idx]) if r.status is ObservationStatus.PASS), None)
    confirmed = defect in first.confirmed
    ambiguous_first = defect in first.ambiguous

    checks = {
        "previous_stage_inspected": prev is not None,
        "previous_stage_passed": prev is not None and prev.status is ObservationStatus.PASS,
        "previous_stage_not_review": prev is not None and prev.status is not ObservationStatus.REVIEW,
        "no_missing_intermediate_stages": not missing,
        "defect_identity_unambiguous": not ambiguous_first,
        "timestamps_consistent": not conflicts,
        "first_observation_confirmed": confirmed,
    }

    if conflicts:
        cls, code = Classification.UNCERTAIN, E.TIMESTAMP_ORDER_CONFLICT
    elif first.stage is FULFILLMENT_FLOW[0]:
        cls, code = Classification.PRE_EXISTING, E.DEFECT_PREEXISTING
    elif prev is None:
        cls, code = Classification.INSUFFICIENT_EVIDENCE, E.NO_PRIOR_OBSERVATION
    elif prev.status is ObservationStatus.REVIEW:
        cls, code = Classification.UNCERTAIN, E.PREVIOUS_STAGE_REVIEW
    elif ambiguous_first:
        cls, code = Classification.UNCERTAIN, E.AMBIGUOUS_DEFECT_IDENTITY
    elif missing:
        cls, code = Classification.INSUFFICIENT_EVIDENCE, E.MISSING_INTERMEDIATE_OBSERVATIONS
    else:
        cls = Classification.NEW_DEFECT
        code = (E.NEW_DEFECT_AFTER_GOOD_STAGE if prev.status is ObservationStatus.PASS
                else E.NEW_DEFECT_AFTER_DEFECTIVE_STAGE)

    if cls is Classification.NEW_DEFECT:
        gaps = (not checks["previous_stage_passed"]) + (not confirmed)
        evidence = (EvidenceLevel.HIGH, EvidenceLevel.MEDIUM, EvidenceLevel.LOW)[gaps]
    elif cls is Classification.PRE_EXISTING:
        evidence = EvidenceLevel.HIGH if confirmed else EvidenceLevel.MEDIUM
    elif cls is Classification.UNCERTAIN:
        evidence = EvidenceLevel.LOW
    else:
        evidence = EvidenceLevel.INSUFFICIENT

    if cls is Classification.NEW_DEFECT:
        attribution = (AttributionStatus.ATTRIBUTED if evidence in (EvidenceLevel.HIGH, EvidenceLevel.MEDIUM)
                       else AttributionStatus.REQUIRES_REVIEW)
    elif cls is Classification.PRE_EXISTING:
        attribution = AttributionStatus.NOT_ATTRIBUTED
    elif cls is Classification.INSUFFICIENT_EVIDENCE:
        attribution = AttributionStatus.INSUFFICIENT_EVIDENCE
    else:
        attribution = AttributionStatus.REQUIRES_REVIEW

    stage_statuses: list[DefectStageEntry] = []
    for i, rec in enumerate(records):
        if i < idx:
            st = DefectStageStatus.UNCERTAIN if rec.status is ObservationStatus.REVIEW else DefectStageStatus.NOT_PRESENT
        elif i == idx:
            st = DefectStageStatus.FIRST_OBSERVED
        elif defect in rec.defects:
            st = DefectStageStatus.PERSISTING
        elif rec.status is ObservationStatus.REVIEW or defect in rec.ambiguous:
            st = DefectStageStatus.UNCERTAIN
        else:
            st = DefectStageStatus.RESOLVED
        stage_statuses.append(DefectStageEntry(stage=rec.stage, status=st))
    final_state = stage_statuses[-1].status
    by_stage = {e.stage: e.status for e in stage_statuses}
    current_status = by_stage.get(current_stage)

    present_now = current_status in (DefectStageStatus.FIRST_OBSERVED, DefectStageStatus.PERSISTING)
    if not present_now:
        current_attr = CurrentStageAttribution.NOT_APPLICABLE
    elif cls is Classification.PRE_EXISTING or current_status is DefectStageStatus.PERSISTING:
        current_attr = CurrentStageAttribution.NOT_SUPPORTED
    elif attribution is AttributionStatus.ATTRIBUTED:
        current_attr = CurrentStageAttribution.SUPPORTED
    else:
        current_attr = CurrentStageAttribution.UNDETERMINED

    flags: list[ExplanationCode] = []
    if not confirmed:
        flags.append(E.FIRST_OBSERVATION_UNCONFIRMED)
    if missing and code is not E.MISSING_INTERMEDIATE_OBSERVATIONS and cls is not Classification.PRE_EXISTING:
        flags.append(E.MISSING_INTERMEDIATE_OBSERVATIONS)
    if prev is not None and prev.status is ObservationStatus.REVIEW and code is not E.PREVIOUS_STAGE_REVIEW:
        flags.append(E.PREVIOUS_STAGE_REVIEW)
    if any(defect in r.ambiguous for r in records) and code is not E.AMBIGUOUS_DEFECT_IDENTITY:
        flags.append(E.AMBIGUOUS_DEFECT_IDENTITY)
    if any(s.status is DefectStageStatus.PERSISTING for s in stage_statuses):
        flags.append(E.DEFECT_PERSISTING)
    if final_state is DefectStageStatus.RESOLVED:
        flags.append(E.DEFECT_RESOLVED)

    return DefectAnalysis(
        defect_type=defect, classification=cls, attribution_status=attribution,
        attributed_stage=first.stage if attribution is AttributionStatus.ATTRIBUTED else None,
        evidence_level=evidence, evidence_checks=checks, first_observed_stage=first.stage,
        first_observed_inspection_id=next(o.inspection_id for o in first.observations if defect in o.defect_types),
        previous_inspected_stage=prev.stage if prev else None,
        previous_stage_status=_timeline_status(prev.status) if prev else None,
        last_known_good_stage=last_good, missing_stages=missing, stage_statuses=stage_statuses,
        final_state=final_state, current_stage_status=current_status, current_stage_attribution=current_attr,
        requires_review=attribution is AttributionStatus.REQUIRES_REVIEW, explanation_code=code, flags=flags)


_STATUS_PRIORITY = (AttributionStatus.REQUIRES_REVIEW, AttributionStatus.INSUFFICIENT_EVIDENCE,
                    AttributionStatus.ATTRIBUTED, AttributionStatus.NOT_ATTRIBUTED)


def analyze_trace(trace: TraceKey, observations: Iterable[PipelineObservation], *,
                  current_inspection_id: str, current_stage: FulfillmentStage) -> FulfillmentTraceAnalysis:
    observations = list(observations)
    for o in observations:
        if not _belongs(o, trace):
            raise TraceMismatchError(f"observation {o.inspection_id} does not belong to trace {trace.key_type.value}")
    kept, duplicates = _dedupe(observations)
    records = _stage_records(kept)
    conflicts = _timestamp_conflicts(records)

    flags: list[ExplanationCode] = []
    if trace.key_type is TraceKeyType.NONE:
        flags.append(E.NO_TRACE_IDENTIFIER)
    if duplicates:
        flags.append(E.DUPLICATE_OBSERVATIONS_IGNORED)
    if conflicts:
        flags.append(E.TIMESTAMP_ORDER_CONFLICT)
    if len(records) < 2:
        flags.append(E.INSUFFICIENT_OBSERVATIONS)

    first_seen: dict[DefectType, int] = {}
    for i, rec in enumerate(records):
        for d in rec.defects:
            first_seen.setdefault(d, i)
    analyses = [_analyse_defect(d, records, bool(conflicts), current_stage)
                for d in sorted(first_seen, key=lambda d: (first_seen[d], d.value))]

    by_stage = {r.stage: r for r in records}
    timeline = []
    for stage in FULFILLMENT_FLOW:
        rec = by_stage.get(stage)
        timeline.append(TimelineEntry(
            stage=stage, position=STAGE_ORDER[stage],
            status=_timeline_status(rec.status) if rec else TimelineStatus.NOT_OBSERVED,
            inspection_ids=[o.inspection_id for o in rec.observations] if rec else [],
            observed_at=rec.last_at if rec else None,
            defects=sorted(rec.defects, key=lambda d: d.value) if rec else []))

    inspected = [r.stage for r in records]
    trace_missing = ([s for s in FULFILLMENT_FLOW[STAGE_ORDER[inspected[0]] - 1: STAGE_ORDER[inspected[-1]]]
                      if s not in by_stage] if inspected else [])

    if not analyses:
        status = AttributionStatus.NOT_ATTRIBUTED
        code = E.NO_DEFECT_OBSERVED if records else E.INSUFFICIENT_OBSERVATIONS
    else:
        status = next(s for s in _STATUS_PRIORITY if any(a.attribution_status is s for a in analyses))
        code = next(a.explanation_code for a in analyses if a.attribution_status is status)

    return FulfillmentTraceAnalysis(
        analyzer_version=ANALYZER_VERSION, trace=trace, current_inspection_id=current_inspection_id,
        current_stage=current_stage, status=status, explanation_code=code,
        requires_review=any(a.requires_review for a in analyses), inspected_stages=inspected,
        missing_stages=trace_missing, timeline=timeline, defect_analyses=analyses, observations=kept,
        duplicates_ignored=duplicates, flags=flags)
