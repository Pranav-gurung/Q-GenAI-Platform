"""Build the RCA evidence payload and the deterministic facts / limitations from the Part 3 trace analysis.

Everything here is derived from `FulfillmentTraceAnalysis` (and optional operator notes). Nothing is re-inferred,
nothing is assumed about uninspected stages, and REVIEW is never rendered as PASS.
"""
from __future__ import annotations

import re

from backend.domain.fulfillment import FULFILLMENT_FLOW, STAGE_ORDER, FulfillmentStage
from backend.schemas.fulfillment import (Classification, DefectAnalysis, DefectStageStatus, ExplanationCode,
                                         FulfillmentTraceAnalysis, ObservationStatus, PipelineObservation,
                                         TimelineStatus)
from backend.schemas.rca import (CAUSALITY_DISCLAIMER, DefectFacts, EvidenceAttribution, EvidenceDefect,
                                 EvidenceObservation, EvidenceTimelineEntry, OperatorNote, RCAEvidence)

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def candidate_stages(a: DefectAnalysis) -> list[FulfillmentStage]:
    """Stages where the defect could have arisen: (last known good, first observed]. From the start of the flow
    when nothing earlier passed. PRE_EXISTING -> [RECEIVING] (present on arrival)."""
    start = STAGE_ORDER[a.last_known_good_stage] if a.last_known_good_stage else 0
    return [s for s in FULFILLMENT_FLOW if start < STAGE_ORDER[s] <= STAGE_ORDER[a.first_observed_stage]]


def sanitize_note(text: str) -> str:
    return _CONTROL.sub(" ", text).strip()[:300]


def _defects(o: PipelineObservation) -> list[EvidenceDefect]:
    return [EvidenceDefect(defect_type=d.defect_type, severity=d.severity, cv_confidence=d.confidence, source=d.source)
            for d in o.defects]


def build_evidence(trace: FulfillmentTraceAnalysis, operator_notes: list[OperatorNote] | None = None) -> RCAEvidence:
    current = next((o for o in trace.observations if o.inspection_id == trace.current_inspection_id), None)
    return RCAEvidence(
        inspection_id=trace.current_inspection_id, store_id=trace.trace.store_id, sku_id=trace.trace.sku_id,
        item_id=trace.trace.item_id, order_id=trace.trace.order_id, trace_key_type=trace.trace.key_type,
        current_stage=trace.current_stage,
        current_decision=current.decision if current else None,
        current_decision_source=current.decision_source if current else None,
        current_defects=_defects(current) if current else [],
        timeline=[EvidenceTimelineEntry(stage=t.stage, status=t.status, defects=t.defects) for t in trace.timeline],
        observations=[EvidenceObservation(inspection_id=o.inspection_id, stage=o.stage, status=o.status,
                                          decision_source=o.decision_source, defects=_defects(o))
                      for o in trace.observations],
        defect_analyses=[EvidenceAttribution(
            defect_type=a.defect_type, classification=a.classification, attribution_status=a.attribution_status,
            attributed_stage=a.attributed_stage, evidence_level=a.evidence_level,
            first_observed_stage=a.first_observed_stage, previous_inspected_stage=a.previous_inspected_stage,
            previous_stage_status=a.previous_stage_status, last_known_good_stage=a.last_known_good_stage,
            missing_stages=a.missing_stages, current_stage_status=a.current_stage_status,
            current_stage_attribution=a.current_stage_attribution, final_state=a.final_state,
            requires_review=a.requires_review, explanation_code=a.explanation_code, flags=a.flags,
            candidate_stages=candidate_stages(a)) for a in trace.defect_analyses],
        trace_status=trace.status, trace_requires_review=trace.requires_review, trace_flags=trace.flags,
        untrusted_operator_notes=[n.model_copy(update={"note": sanitize_note(n.note)}) for n in (operator_notes or [])
                                  if sanitize_note(n.note)])


# ------------------------------------------------------------------------------------------------ deterministic text
def _stages(stages) -> str:
    return ", ".join(s.value for s in stages)


def _first_model_defect(trace: FulfillmentTraceAnalysis, a: DefectAnalysis):
    obs = next(o for o in trace.observations if o.inspection_id == a.first_observed_inspection_id)
    return next((d for d in obs.defects if d.defect_type is a.defect_type), None)


def defect_facts(trace: FulfillmentTraceAnalysis, a: DefectAnalysis) -> DefectFacts:
    d = _first_model_defect(trace, a)
    return DefectFacts(
        first_observed_stage=a.first_observed_stage, last_known_good_stage=a.last_known_good_stage,
        previous_inspected_stage=a.previous_inspected_stage, previous_stage_status=a.previous_stage_status,
        missing_stages=a.missing_stages, classification=a.classification, evidence_level=a.evidence_level,
        attribution_status=a.attribution_status, attributed_stage=a.attributed_stage,
        current_stage_attribution=a.current_stage_attribution, requires_review=a.requires_review,
        explanation_code=a.explanation_code, cv_confidence=d.confidence if d else None,
        severity=d.severity if d else None)


def timeline_sentence(trace: FulfillmentTraceAnalysis) -> str:
    parts = []
    for t in trace.timeline:
        label = "not inspected" if t.status is TimelineStatus.NOT_OBSERVED else t.status.value
        if t.defects:
            label += f" ({', '.join(d.value for d in t.defects)})"
        parts.append(f"{t.stage.value}: {label}")
    return "Inspection timeline - " + "; ".join(parts) + "."


def observed_evidence(trace: FulfillmentTraceAnalysis, a: DefectAnalysis) -> list[str]:
    D, first = a.defect_type.value, a.first_observed_stage.value
    out = []
    if a.classification is Classification.PRE_EXISTING:
        out.append(f"{D} was already observed at RECEIVING, the first inspection of this item (present on arrival).")
    else:
        out.append(f"{D} was first observed at {first}.")
    if a.last_known_good_stage:
        out.append(f"The last inspection that explicitly passed was {a.last_known_good_stage.value}.")
    else:
        out.append("No earlier inspection of this item explicitly passed.")
    if a.previous_inspected_stage and a.previous_stage_status is TimelineStatus.REVIEW:
        out.append(f"The previous inspection ({a.previous_inspected_stage.value}) required review; it is not treated "
                   "as a pass.")
    elif a.previous_inspected_stage and a.previous_stage_status is TimelineStatus.FAIL:
        out.append(f"The previous inspection ({a.previous_inspected_stage.value}) failed for a different defect.")
    if a.missing_stages and a.classification is not Classification.PRE_EXISTING:
        out.append(f"Not inspected before the first observation: {_stages(a.missing_stages)}.")
    d = _first_model_defect(trace, a)
    if d and d.confidence is not None:
        out.append(f"CV detection at {first}: confidence {d.confidence:.2f}, severity "
                   f"{d.severity.value if d.severity else 'n/a'}.")
    elif d and d.source.value == "OPERATOR":
        out.append(f"The defect at {first} was recorded by an operator.")
    later = [s for s in a.stage_statuses if STAGE_ORDER[s.stage] > STAGE_ORDER[a.first_observed_stage]]
    persisting = [s.stage for s in later if s.status is DefectStageStatus.PERSISTING]
    resolved = [s.stage for s in later if s.status is DefectStageStatus.RESOLVED]
    if persisting:
        out.append(f"Still observed at {_stages(persisting)} (persistence of the same defect type, not a new defect).")
    if resolved:
        out.append(f"Not observed at {_stages(resolved)} (resolved, replaced, or not detected).")
    out.append(f"Stage attribution: {a.classification.value}, evidence {a.evidence_level.value} "
               f"({a.explanation_code.value}).")
    return out


def limitations(trace: FulfillmentTraceAnalysis, a: DefectAnalysis) -> list[str]:
    out = [CAUSALITY_DISCLAIMER]
    E = ExplanationCode
    if a.classification is Classification.PRE_EXISTING:
        out.append("The defect was present at the first inspection; the evidence does not support attributing it to "
                   "any later stage.")
    if a.explanation_code is E.MISSING_INTERMEDIATE_OBSERVATIONS or (
            a.missing_stages and a.classification is not Classification.PRE_EXISTING):
        out.append(f"{_stages(a.missing_stages)} were not inspected, so the available evidence does not establish "
                   "when the defect physically occurred.")
    if a.explanation_code is E.NO_PRIOR_OBSERVATION:
        out.append("There is no earlier inspection of this item, so the stage where the defect arose is unknown.")
    if a.explanation_code is E.PREVIOUS_STAGE_REVIEW or E.PREVIOUS_STAGE_REVIEW in a.flags:
        out.append("An earlier inspection required review and is not counted as a pass.")
    if a.explanation_code is E.AMBIGUOUS_DEFECT_IDENTITY or E.AMBIGUOUS_DEFECT_IDENTITY in a.flags:
        out.append("Another defect type disappeared when this one appeared; it may be a re-classification of the same "
                   "damage.")
    if a.explanation_code is E.TIMESTAMP_ORDER_CONFLICT:
        out.append("Inspection timestamps are out of stage order; the sequence of events is uncertain.")
    if E.FIRST_OBSERVATION_UNCONFIRMED in a.flags:
        out.append("The first observation is an unconfirmed (review-level) detection.")
    if a.requires_review:
        out.append("Evidence is insufficient to confidently attribute the defect to a specific fulfillment stage; a "
                   "human review is required.")
    return out


def trace_limitations(trace: FulfillmentTraceAnalysis) -> list[str]:
    out = []
    if trace.trace.key_type.value == "NONE":
        out.append("This inspection has no item or order id, so no other stage evidence could be linked.")
    if any(o.status is ObservationStatus.REVIEW for o in trace.observations):
        out.append("Some inspections required review; they are not treated as passes.")
    return out
