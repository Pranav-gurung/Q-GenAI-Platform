"""Feedback error analysis + learning-candidate export (Part 6). Read-only, deterministic, no ML.

Uses the LATEST feedback per inspection (earlier submissions stay stored, but are superseded for statistics).
Decision overrides count every feedback on a real automated decision (old records too); defect / stage / RCA /
action statistics count only the structured Part 6 reviews.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.db.models import Feedback, Inspection
from backend.domain.enums import DecisionOutcome
from backend.schemas.common import utcnow
from backend.schemas.learning import (BreakdownStats, FeedbackAnalysis, LayerStats, LearningCandidate,
                                      UsefulnessStats)
from backend.services.mappers import feedback_corrections


@dataclass(frozen=True)
class FeedbackFilters:
    store_id: str | None = None
    sku_id: str | None = None
    created_from: datetime | None = None   # inspection created_at >= (Part 7 analytics period)
    created_before: datetime | None = None  # inspection created_at <


def _value(x):
    return getattr(x, "value", x)


class LearningService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _latest(self, filters: FeedbackFilters) -> tuple[list[Feedback], int]:
        q = (select(Feedback).join(Inspection, Feedback.inspection_id == Inspection.inspection_id)
             .options(selectinload(Feedback.inspection).selectinload(Inspection.images))
             .order_by(Feedback.created_at, Feedback.feedback_id))
        if filters.store_id:
            q = q.where(Inspection.store_id == filters.store_id)
        if filters.sku_id:
            q = q.where(Inspection.sku_id == filters.sku_id)
        if filters.created_from:
            q = q.where(Inspection.created_at >= filters.created_from)
        if filters.created_before:
            q = q.where(Inspection.created_at < filters.created_before)
        rows = list(self.session.scalars(q))
        latest = {fb.inspection_id: fb for fb in rows}  # ascending -> last wins
        return sorted(latest.values(), key=lambda f: (f.created_at, f.feedback_id)), len(rows)

    # ------------------------------------------------------------------------------------------------ analysis
    def analyze(self, filters: FeedbackFilters = FeedbackFilters()) -> FeedbackAnalysis:
        latest, submissions = self._latest(filters)
        decisions = _Breakdown()
        defects, stages = _Breakdown(), _Breakdown()
        rca, actions, sop_actions = _Usefulness(), _Usefulness(), _Usefulness()
        for fb in latest:
            if fb.system_outcome is not None:
                decisions.add([_value(fb.system_outcome)], not fb.is_correction,
                              f"{_value(fb.system_outcome)} -> {_value(fb.outcome)}")
            if fb.defect_correct is not None:
                ai = list(fb.ai_defect_types or []) or ["NONE"]
                defects.add(ai, fb.defect_correct,
                            f"{'+'.join(ai)} -> {'+'.join(fb.defect_types or []) or 'NONE'}")
            if fb.stage_correct is not None:
                stages.add([_value(fb.ai_stage)], fb.stage_correct,
                           f"{_value(fb.ai_stage)} -> {_value(fb.suspected_stage)}")
            rca.add(fb.rca_useful)
            actions.add(fb.action_useful)
            if ((fb.ai_snapshot or {}).get("rag") or {}).get("status") == "GROUNDED":
                sop_actions.add(fb.action_useful)  # actions the operator saw were SOP-grounded (Part 5)
        return FeedbackAnalysis(
            generated_at=utcnow(), feedback_submissions=submissions, inspections_reviewed=len(latest),
            decisions=decisions.stats(), defects=defects.stats(), stages=stages.stats(), rca=rca.stats(),
            actions=actions.stats(), sop_grounded_actions=sop_actions.stats(),
            learning_status=dict(Counter(_value(fb.learning_status) for fb in latest)))

    # ------------------------------------------------------------------------------------------------ export
    def candidates(self, filters: FeedbackFilters = FeedbackFilters(), *, corrections_only: bool = False,
                   limit: int = 500) -> list[LearningCandidate]:
        latest, _ = self._latest(filters)
        out = []
        for fb in latest:
            layers = [c.layer for c in feedback_corrections(fb)]
            if corrections_only and not (layers or fb.is_correction):
                continue
            if fb.is_correction and "DECISION" not in layers:  # override recorded without the Part 6 flag
                layers.insert(0, "DECISION")
            insp, img = fb.inspection, fb.inspection.current_image
            labelled = fb.defect_types or fb.defect_correct is not None or fb.outcome is DecisionOutcome.PASS
            out.append(LearningCandidate(
                feedback_id=fb.feedback_id, inspection_id=fb.inspection_id, item_id=insp.item_id,
                order_id=insp.order_id, store_id=insp.store_id, sku_id=insp.sku_id, inspection_stage=insp.stage,
                image_id=img.image_id if img else None, image_sha256=img.sha256 if img else None,
                ai_decision=fb.system_outcome, operator_decision=fb.outcome, ai_defects=fb.ai_defect_types,
                operator_defects=list(fb.defect_types or []) if labelled else None, ai_stage=fb.ai_stage,
                operator_stage=fb.suspected_stage, decision_correct=fb.decision_correct,
                defect_correct=fb.defect_correct, stage_correct=fb.stage_correct, rca_useful=fb.rca_useful,
                action_useful=fb.action_useful, correction_layers=layers, comment=fb.comment,
                model_versions=dict(fb.model_versions or {}), learning_status=fb.learning_status,
                created_at=fb.created_at))
            if len(out) >= limit:
                break
        return out


class _Breakdown:
    def __init__(self) -> None:
        self.total = LayerStats()
        self.by: dict[str, LayerStats] = {}
        self.transitions: Counter = Counter()

    def add(self, keys: list[str], correct: bool, transition: str) -> None:
        for stats in [self.total, *(self.by.setdefault(k, LayerStats()) for k in keys)]:
            stats.reviewed += 1
            stats.confirmed += correct
            stats.corrected += not correct
        if not correct:
            self.transitions[transition] += 1

    def stats(self) -> BreakdownStats:
        for s in [self.total, *self.by.values()]:
            s.correction_rate = round(s.corrected / s.reviewed, 4) if s.reviewed else None
        return BreakdownStats(**self.total.model_dump(), by_ai_value=dict(sorted(self.by.items())),
                              transitions=dict(sorted(self.transitions.items())))


class _Usefulness:
    def __init__(self) -> None:
        self.useful = self.not_useful = 0

    def add(self, value: bool | None) -> None:
        if value is True:
            self.useful += 1
        elif value is False:
            self.not_useful += 1

    def stats(self) -> UsefulnessStats:
        rated = self.useful + self.not_useful
        return UsefulnessStats(rated=rated, useful=self.useful, not_useful=self.not_useful,
                               useful_rate=round(self.useful / rated, 4) if rated else None)
