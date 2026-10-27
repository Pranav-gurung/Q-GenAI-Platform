"""Operations intelligence (Part 7): deterministic quality analytics over stored inspections and feedback.

Read-only. It reuses the existing semantics instead of re-implementing them:
  * per-inspection outcome and observed defects -> Part 3 `build_observation` (operator verdict over rules decision),
  * first observed stage per item                -> Part 3 `analyze_trace` on the period's observations,
  * operator feedback rates                      -> Part 6 `LearningService.analyze`.

Database work: the period bounds, in-flight counts and reference names are SQL aggregates; per-inspection data is
fetched as plain columns in three parameterised queries (inspections + decision/CV/quality JSON, latest image, latest
feedback), never as full ORM graphs, and aggregated in one pass. Defects live inside JSON result documents and the
operator-precedence rule is Python, so a portable (SQLite + PostgreSQL) pure-SQL version would duplicate Part 3 logic;
the period is capped (MAX_PERIOD_DAYS) to keep the scan bounded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.core.exceptions import ValidationFailedError
from backend.db.models import Feedback, Inspection, InspectionImage, InspectionResult, Sku, Store
from backend.domain.enums import DecisionOutcome, DefectType, ModuleStatus
from backend.domain.fulfillment import FULFILLMENT_FLOW, FulfillmentStage
from backend.fulfillment.analyzer import analyze_trace
from backend.fulfillment.observations import build_observation, trace_key_for
from backend.pipeline.states import PipelineState
from backend.schemas.analytics import (AnalyticsOverview, DefectStageMatrix, DefectTypeMetrics, FeedbackMetrics,
                                       GroupMetrics, Hotspot, InspectionMetrics, Investigation, MatrixRow, Period,
                                       StageMetrics, TrendBucket, TrendDirection, Trends)
from backend.schemas.fulfillment import EvidenceSource, PipelineObservation, TraceKeyType
from backend.schemas.results import CVResult, QualityDecision
from backend.services.learning_service import FeedbackFilters, LearningService

REVIEWABLE = (PipelineState.COMPLETED, PipelineState.REQUIRES_REVIEW, PipelineState.FAILED)
DEFAULT_PERIOD_DAYS = 30
MAX_PERIOD_DAYS = 366
TREND_THRESHOLD = 0.05
DEMO_SEED_KEY = "demo_seed"  # inspections.metadata marker written by scripts/seed_demo_operations.py


@dataclass(frozen=True)
class AnalyticsFilters:
    start_date: date | None = None
    end_date: date | None = None
    store_id: str | None = None
    sku_id: str | None = None
    granularity: str | None = None  # day | week; default: day up to 31 days, else week


@dataclass
class _Row:
    inspection_id: str
    store_id: str
    sku_id: str
    stage: FulfillmentStage
    created: date
    outcome: str  # PASS | FAIL | REVIEW | IMAGE_REJECTED | PIPELINE_FAILED | NO_DECISION
    operator: bool
    observation: PipelineObservation
    severities: dict[DefectType, str] = field(default_factory=dict)

    @property
    def decided(self) -> bool:
        return self.outcome in ("PASS", "FAIL", "REVIEW")

    @property
    def defect_types(self) -> list[DefectType]:
        return [d.defect_type for d in self.observation.defects] if self.decided else []


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _utc(d: date) -> datetime:
    return datetime.combine(d, time.min, tzinfo=timezone.utc)


def _day(ts: datetime) -> date:
    return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).date()


class AnalyticsService:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------------------------------------ period
    def _scope(self, q, f: AnalyticsFilters):
        if f.store_id:
            q = q.where(Inspection.store_id == f.store_id)
        if f.sku_id:
            q = q.where(Inspection.sku_id == f.sku_id)
        return q

    def _period(self, f: AnalyticsFilters) -> Period | None:
        start, end = f.start_date, f.end_date
        defaulted = start is None and end is None
        if end is None:  # default end: the latest inspection's day (never before start_date)
            latest = self.session.scalar(self._scope(select(func.max(Inspection.created_at)), f))
            if latest is None and start is None:
                return None
            end = _day(latest) if latest is not None else start
            if start is not None and end < start:
                end = start
        if start is None:  # default start: up to 30 days back, but not before the first inspection in that window
            start = end - timedelta(days=DEFAULT_PERIOD_DAYS - 1)
            first = self.session.scalar(self._scope(select(func.min(Inspection.created_at)), f)
                                        .where(Inspection.created_at >= _utc(start)))
            if first is not None and _day(first) <= end:
                start = max(start, _day(first))
        if start > end:
            raise ValidationFailedError("start_date must be on or before end_date", code="INVALID_DATE_RANGE",
                                        details={"start_date": str(start), "end_date": str(end)})
        days = (end - start).days + 1
        if days > MAX_PERIOD_DAYS:
            raise ValidationFailedError(f"The period may span at most {MAX_PERIOD_DAYS} days",
                                        code="INVALID_DATE_RANGE", details={"days": days})
        granularity = f.granularity or ("day" if days <= 31 else "week")
        return Period(start=start, end=end, days=days, granularity=granularity, defaulted=defaulted)

    # ------------------------------------------------------------------------------------------------ load
    def _load(self, f: AnalyticsFilters, p: Period) -> tuple[list[_Row], int, int]:
        lo, hi = _utc(p.start), _utc(p.end + timedelta(days=1))
        in_period = self._scope(select(Inspection.inspection_id), f).where(Inspection.created_at >= lo,
                                                                          Inspection.created_at < hi)
        in_progress = self.session.scalar(select(func.count()).select_from(
            in_period.where(Inspection.status.not_in(REVIEWABLE)).subquery())) or 0
        ids = in_period.where(Inspection.status.in_(REVIEWABLE))

        rows = self.session.execute(
            select(Inspection.inspection_id, Inspection.store_id, Inspection.sku_id, Inspection.item_id,
                   Inspection.order_id, Inspection.stage, Inspection.status, Inspection.created_at,
                   Inspection.metadata_, InspectionResult.decision, InspectionResult.cv, InspectionResult.quality)
            .outerjoin(InspectionResult, InspectionResult.inspection_id == Inspection.inspection_id)
            .where(Inspection.inspection_id.in_(ids))
            .order_by(Inspection.created_at, Inspection.inspection_id)).all()
        images = {r.inspection_id: r for r in self.session.execute(
            select(InspectionImage.inspection_id, InspectionImage.image_id, InspectionImage.sha256,
                   InspectionImage.created_at).where(InspectionImage.inspection_id.in_(ids))
            .order_by(InspectionImage.created_at, InspectionImage.image_id))}  # last image wins
        feedback = {r.inspection_id: r for r in self.session.execute(
            select(Feedback.inspection_id, Feedback.outcome, Feedback.defect_types)
            .where(Feedback.inspection_id.in_(ids)).order_by(Feedback.created_at, Feedback.feedback_id))}

        out, demo = [], 0
        for r in rows:
            demo += bool((r.metadata_ or {}).get(DEMO_SEED_KEY))
            img, fb = images.get(r.inspection_id), feedback.get(r.inspection_id)
            decision = QualityDecision.model_validate(r.decision) if r.decision else None
            cv = CVResult.model_validate(r.cv) if r.cv else None
            obs = build_observation(
                inspection_id=r.inspection_id, stage=r.stage, observed_at=img.created_at if img else r.created_at,
                store_id=r.store_id, sku_id=r.sku_id, item_id=r.item_id, order_id=r.order_id, decision=decision,
                cv=cv, operator_outcome=DecisionOutcome(fb.outcome) if fb else None,
                operator_defects=[DefectType(t) for t in (fb.defect_types or [])] if fb else None,
                image_id=img.image_id if img else None, image_sha256=img.sha256 if img else None)
            if obs.decision_source is not EvidenceSource.NO_DECISION:
                outcome = obs.status.value
            elif (r.quality or {}).get("status") == ModuleStatus.COMPLETED.value and \
                    (r.quality or {}).get("acceptable") is False:
                outcome = "IMAGE_REJECTED"
            elif r.status is PipelineState.FAILED:
                outcome = "PIPELINE_FAILED"
            else:
                outcome = "NO_DECISION"
            # severity: CV's max severity for that type when the model detected it (also when an operator confirmed
            # it); UNSPECIFIED for operator-only labels
            cv_sev = {d.defect_type: d.severity for d in build_observation(
                inspection_id=r.inspection_id, stage=r.stage, observed_at=r.created_at, store_id=r.store_id,
                sku_id=r.sku_id, item_id=None, order_id=None, decision=None, cv=cv).defects}
            out.append(_Row(r.inspection_id, r.store_id, r.sku_id, r.stage, _day(r.created_at), outcome,
                            obs.decision_source is EvidenceSource.OPERATOR, obs,
                            {d.defect_type: (sev.value if (sev := d.severity or cv_sev.get(d.defect_type))
                                             else "UNSPECIFIED") for d in obs.defects}))
        return out, in_progress, demo

    # ------------------------------------------------------------------------------------------------ overview
    def overview(self, f: AnalyticsFilters) -> AnalyticsOverview:
        filters = {"store_id": f.store_id, "sku_id": f.sku_id}
        p = self._period(f)
        if p is None:
            return _empty(None, filters, "No inspection data available.")
        rows, in_progress, demo = self._load(f, p)
        fb = self._feedback(f, p)
        if not rows:
            out = _empty(p, filters, "No inspection data available for this period.")
            out.inspections.in_progress = in_progress
            out.feedback = fb
            return out

        decided = [r for r in rows if r.decided]
        firsts = _first_observations(rows)  # (trace, defect) -> (first stage, evidence level, store, sku)
        unique_total = len(firsts)

        counts = Counter(r.outcome for r in rows)
        defective = sum(1 for r in decided if r.defect_types)
        inspections = InspectionMetrics(
            total=len(rows), in_progress=in_progress, decided=len(decided), passed=counts["PASS"],
            fail=counts["FAIL"], review=counts["REVIEW"], image_rejected=counts["IMAGE_REJECTED"],
            pipeline_failed=counts["PIPELINE_FAILED"], no_decision=counts["NO_DECISION"],
            operator_verdicts=sum(1 for r in decided if r.operator), defective=defective,
            fail_rate=_rate(counts["FAIL"], len(decided)), observed_defect_rate=_rate(defective, len(decided)))

        # defects by type (top issues)
        obs_by_type: Counter = Counter()
        sev_by_type: dict[DefectType, Counter] = defaultdict(Counter)
        for r in decided:
            for t in r.defect_types:
                obs_by_type[t] += 1
                sev_by_type[t][r.severities.get(t, "UNSPECIFIED")] += 1
        unique_by_type: Counter = Counter(t for (_, t) in firsts)
        stage_by_type: dict[DefectType, Counter] = defaultdict(Counter)
        for (_, t), v in firsts.items():
            stage_by_type[t][v["stage"]] += 1
        defects = [DefectTypeMetrics(
            defect_type=t, observations=obs_by_type[t], unique_defects=unique_by_type[t],
            share_of_unique=_rate(unique_by_type[t], unique_total),
            most_common_first_stage=_top(stage_by_type[t], key=FULFILLMENT_FLOW.index),
            severity=dict(sorted(sev_by_type[t].items()))) for t in set(obs_by_type) | set(unique_by_type)]
        defects.sort(key=lambda d: (-d.unique_defects, -d.observations, d.defect_type.value))

        # stages
        first_by_stage: Counter = Counter(v["stage"] for v in firsts.values())
        stages = []
        for s in FULFILLMENT_FLOW:
            at = [r for r in decided if r.stage is s]
            if not at and not first_by_stage[s]:
                continue
            d = sum(1 for r in at if r.defect_types)
            stages.append(StageMetrics(stage=s, inspections=len(at), defective_inspections=d,
                                       observed_defect_rate=_rate(d, len(at)),
                                       defect_observations=sum(len(r.defect_types) for r in at),
                                       first_observed_defects=first_by_stage[s]))
        rate_of = {m.stage: m.observed_defect_rate for m in stages}
        ranked = sorted((s for s in FULFILLMENT_FLOW if first_by_stage[s]),
                        key=lambda s: (-first_by_stage[s], -(rate_of.get(s) or 0), FULFILLMENT_FLOW.index(s)))
        hotspots = []
        for i, s in enumerate(ranked, 1):
            here = [v for v in firsts.values() if v["stage"] is s]
            hotspots.append(Hotspot(
                rank=i, stage=s, first_observed_defects=first_by_stage[s], share=_rate(first_by_stage[s], unique_total),
                observed_defect_rate=rate_of.get(s),
                top_defect_type=_top(Counter(t for (_, t), v in firsts.items() if v["stage"] is s),
                                     key=lambda t: t.value),
                evidence_levels=dict(sorted(Counter(v["evidence"] for v in here).items()))))

        inspected = {r.stage for r in decided}
        matrix = DefectStageMatrix(stages=list(FULFILLMENT_FLOW), rows=[
            MatrixRow(defect_type=d.defect_type, total=d.unique_defects,
                      counts={s.value: (stage_by_type[d.defect_type][s] if s in inspected or
                                        stage_by_type[d.defect_type][s] else None) for s in FULFILLMENT_FLOW})
            for d in defects if d.unique_defects])

        names = self._names()
        stores = _groups(decided, firsts, lambda r: r.store_id, lambda v: v["store"], names["stores"])
        skus = _groups(decided, firsts, lambda r: r.sku_id, lambda v: v["sku"], names["skus"])
        cat = {k: (v[1] or "Uncategorised") for k, v in names["skus"].items()}
        categories = _groups(decided, firsts, lambda r: cat.get(r.sku_id, "Uncategorised"),
                             lambda v: cat.get(v["sku"], "Uncategorised"), {})

        return AnalyticsOverview(
            period=p, filters=filters, has_data=bool(decided),
            message=None if decided else "No decided inspections in this period (only rejected or undecided ones).",
            demo_seed_inspections=demo, inspections=inspections, defects=defects, unique_defects=unique_total,
            defect_observations=sum(obs_by_type.values()), stages=stages, hotspots=hotspots, matrix=matrix,
            stores=stores, skus=skus, categories=categories, trends=_trends(decided, p), feedback=fb,
            investigation=_investigation(hotspots, fb, unique_total))

    def _feedback(self, f: AnalyticsFilters, p: Period) -> FeedbackMetrics:
        a = LearningService(self.session).analyze(FeedbackFilters(
            store_id=f.store_id, sku_id=f.sku_id, created_from=_utc(p.start),
            created_before=_utc(p.end + timedelta(days=1))))
        return FeedbackMetrics(
            feedback_records=a.inspections_reviewed,
            decision_reviews=a.decisions.reviewed, decision_overrides=a.decisions.corrected,
            decision_override_rate=a.decisions.correction_rate,
            defect_reviews=a.defects.reviewed, defect_corrections=a.defects.corrected,
            defect_correction_rate=a.defects.correction_rate,
            stage_reviews=a.stages.reviewed, stage_corrections=a.stages.corrected,
            stage_correction_rate=a.stages.correction_rate,
            stage_corrections_by_ai_stage={k: v.corrected for k, v in a.stages.by_ai_value.items() if v.corrected},
            rca=a.rca, actions=a.actions, sop_grounded_actions=a.sop_grounded_actions)

    def _names(self) -> dict:
        return {"stores": {r.store_id: (r.name, None) for r in self.session.execute(select(Store.store_id, Store.name))},
                "skus": {r.sku_id: (r.name, r.category) for r in
                         self.session.execute(select(Sku.sku_id, Sku.name, Sku.category))}}


# ------------------------------------------------------------------------------------------------ helpers
def _top(counter: Counter, key):
    """Most common value; ties broken deterministically by `key`."""
    if not counter:
        return None
    best = max(counter.values())
    return min((k for k, v in counter.items() if v == best), key=key)


def _first_observations(rows: list[_Row]) -> dict:
    """Part 3 analysis per item trace (period observations only) -> one entry per (trace, defect type)."""
    traces: dict[tuple, list[_Row]] = defaultdict(list)
    for r in rows:
        key = trace_key_for(r.store_id, r.sku_id, r.observation.item_id, r.observation.order_id)
        ident = (key.key_type.value, r.store_id, r.sku_id, key.item_id or key.order_id or r.inspection_id)
        traces[ident].append(r)
    out = {}
    for ident, members in traces.items():
        last = max(members, key=lambda r: (r.observation.observed_at, r.inspection_id))
        key = trace_key_for(last.store_id, last.sku_id, last.observation.item_id, last.observation.order_id)
        if key.key_type is TraceKeyType.NONE:
            members = [last]  # unlinkable inspections are their own trace
        analysis = analyze_trace(key, [m.observation for m in members], current_inspection_id=last.inspection_id,
                                 current_stage=last.stage)
        for a in analysis.defect_analyses:
            out[(ident, a.defect_type)] = {"stage": a.first_observed_stage, "evidence": a.evidence_level.value,
                                           "store": last.store_id, "sku": last.sku_id}
    return out


def _groups(decided: list[_Row], firsts: dict, key_of, first_key_of, names: dict) -> list[GroupMetrics]:
    by: dict[str, list[_Row]] = defaultdict(list)
    for r in decided:
        by[key_of(r)].append(r)
    uniq: dict[str, Counter] = defaultdict(Counter)
    for (_, t), v in firsts.items():
        uniq[first_key_of(v)][t] += 1
    out = []
    for k, rs in by.items():
        fail = sum(1 for r in rs if r.outcome == "FAIL")
        d = sum(1 for r in rs if r.defect_types)
        name, category = names.get(k, (None, None))
        out.append(GroupMetrics(key=k, name=name, category=category, inspections=len(rs), fail=fail,
                                fail_rate=_rate(fail, len(rs)), defective_inspections=d,
                                observed_defect_rate=_rate(d, len(rs)), unique_defects=sum(uniq[k].values()),
                                top_defect_type=_top(uniq[k], key=lambda t: t.value)))
    out.sort(key=lambda g: (-g.unique_defects, -(g.observed_defect_rate or 0), -g.inspections, g.key))
    return out


def _bucket(d: date, granularity: str) -> date:
    return d - timedelta(days=d.weekday()) if granularity == "week" else d


def _trends(decided: list[_Row], p: Period) -> Trends:
    g = p.granularity
    agg: dict[date, list[_Row]] = defaultdict(list)
    for r in decided:
        agg[_bucket(r.created, g)].append(r)
    buckets, cur, step = [], _bucket(p.start, g), timedelta(days=7 if g == "week" else 1)
    while cur <= p.end:
        rs = agg.get(cur, [])
        d = sum(1 for r in rs if r.defect_types)
        buckets.append(TrendBucket(start=cur, inspections=len(rs), fail=sum(1 for r in rs if r.outcome == "FAIL"),
                                   defective_inspections=d, defect_observations=sum(len(r.defect_types) for r in rs),
                                   observed_defect_rate=_rate(d, len(rs))))
        cur += step
    return Trends(granularity=g, buckets=buckets, direction=_direction([b for b in buckets if b.inspections]))


def _direction(filled: list[TrendBucket]) -> TrendDirection:
    if len(filled) < 2:
        return TrendDirection(direction="insufficient_data",
                              statement="Not enough periods with inspections to describe a trend.")
    half = len(filled) // 2
    first, second = filled[:half], filled[len(filled) - half:]  # an odd middle bucket is left out
    r1 = _rate(sum(b.defective_inspections for b in first), sum(b.inspections for b in first))
    r2 = _rate(sum(b.defective_inspections for b in second), sum(b.inspections for b in second))
    delta = (r2 or 0) - (r1 or 0)
    direction = "increased" if delta >= TREND_THRESHOLD else "decreased" if delta <= -TREND_THRESHOLD else "stable"
    statement = {"increased": "The observed defect rate increased", "decreased": "The observed defect rate decreased",
                 "stable": "The observed defect rate was stable"}[direction]
    return TrendDirection(direction=direction, first_half_rate=r1, second_half_rate=r2,
                          statement=f"{statement} between the first and second half of the period "
                                    f"({_pct(r1)} -> {_pct(r2)}). This describes observations, not their cause.")


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _investigation(hotspots: list[Hotspot], fb: FeedbackMetrics, unique_total: int) -> Investigation | None:
    if not hotspots:
        return None
    h = hotspots[0]
    reasons = [f"Highest observed defect volume first observed at this stage in the selected period: "
               f"{h.first_observed_defects} of {unique_total} unique defects ({_pct(h.share)})."]
    if h.top_defect_type:
        reasons.append(f"Most frequent defect first observed here: {h.top_defect_type.value}.")
    if h.observed_defect_rate is not None:
        reasons.append(f"Observed defect rate among inspections at this stage: {_pct(h.observed_defect_rate)}.")
    strong = h.evidence_levels.get("HIGH", 0)
    reasons.append(f"{strong} of {h.first_observed_defects} first observations have HIGH evidence (the previous "
                   "stage was inspected and passed); the rest have weaker evidence.")
    corrected = fb.stage_corrections_by_ai_stage.get(h.stage.value, 0)
    if corrected:
        reasons.append(f"Operators corrected {corrected} stage attribution(s) that pointed at this stage; review "
                       "those inspections before acting.")
    return Investigation(stage=h.stage, headline=f"{h.stage.value} is the highest observed quality hotspot and "
                                                 "should be investigated first.", reasons=reasons)


def _empty(p: Period | None, filters: dict, message: str) -> AnalyticsOverview:
    zero_use = {"rated": 0, "useful": 0, "not_useful": 0, "useful_rate": None}
    return AnalyticsOverview(
        period=p, filters=filters, has_data=False, message=message,
        inspections=InspectionMetrics(total=0, in_progress=0, decided=0, passed=0, fail=0, review=0, image_rejected=0,
                                      pipeline_failed=0, no_decision=0, operator_verdicts=0, defective=0,
                                      fail_rate=None, observed_defect_rate=None),
        defects=[], unique_defects=0, defect_observations=0, stages=[], hotspots=[],
        matrix=DefectStageMatrix(stages=list(FULFILLMENT_FLOW), rows=[]), stores=[], skus=[], categories=[],
        trends=_trends([], p) if p else Trends(granularity="day", buckets=[], direction=_direction([])),
        feedback=FeedbackMetrics(feedback_records=0, decision_reviews=0, decision_overrides=0,
                                 decision_override_rate=None, defect_reviews=0, defect_corrections=0,
                                 defect_correction_rate=None, stage_reviews=0, stage_corrections=0,
                                 stage_correction_rate=None, stage_corrections_by_ai_stage={}, rca=zero_use,
                                 actions=zero_use, sop_grounded_actions=zero_use),
        investigation=None)
