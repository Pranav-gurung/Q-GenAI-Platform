"""Pure deterministic analyzer: the 10 required cases + evidence / edge-case rules."""
import random
import time

import pytest

from backend.domain.fulfillment import FULFILLMENT_FLOW, STAGE_ORDER, stages_between
from backend.fulfillment.analyzer import TraceMismatchError, analyze_trace
from backend.schemas.fulfillment import (AttributionStatus as A, Classification as C, CurrentStageAttribution as CA,
                                         DefectStageStatus as DS, EvidenceLevel as EV, EvidenceSource,
                                         ExplanationCode as X, TimelineStatus as TS, TraceKey, TraceKeyType)
from tests.fulfillment.helpers import CRUSHED, DENT, LABEL, LEAK, TRACE, D, K, P, R, S, V, obs


def run(observations, current=None, trace=TRACE):
    current = current or max(observations, key=lambda o: STAGE_ORDER[o.stage]).stage if observations else K
    return analyze_trace(trace, observations, current_inspection_id="INS-CURRENT", current_stage=current)


def one(result):
    assert len(result.defect_analyses) == 1
    return result.defect_analyses[0]


def timeline(result):
    return {e.stage: e.status for e in result.timeline}


def test_canonical_stage_order_is_single_source():
    assert [STAGE_ORDER[s] for s in FULFILLMENT_FLOW] == [1, 2, 3, 4, 5, 6]
    assert [s.value for s in FULFILLMENT_FLOW] == ["RECEIVING", "STORAGE", "PICKING", "PACKING", "DISPATCH", "DELIVERY"]
    assert stages_between(R, K) == (S, P) and stages_between(P, K) == ()


# ---------------------------------------------------------------- required cases
def test_case1_all_pass_no_defect():
    r = run([obs(R), obs(S), obs(P), obs(K), obs(D)])
    assert r.status is A.NOT_ATTRIBUTED and r.explanation_code is X.NO_DEFECT_OBSERVED
    assert r.defect_analyses == [] and not r.requires_review
    assert timeline(r) == {R: TS.PASS, S: TS.PASS, P: TS.PASS, K: TS.PASS, D: TS.PASS, V: TS.NOT_OBSERVED}


def test_case2_new_defect_at_packing_high_evidence():
    r = run([obs(R), obs(S), obs(P), obs(K, "FAIL", [CRUSHED]), obs(D, "FAIL", [CRUSHED])], current=K)
    a = one(r)
    assert a.classification is C.NEW_DEFECT and a.explanation_code is X.NEW_DEFECT_AFTER_GOOD_STAGE
    assert a.first_observed_stage is K and a.last_known_good_stage is P and a.previous_inspected_stage is P
    assert a.evidence_level is EV.HIGH and all(a.evidence_checks.values())
    assert a.attribution_status is A.ATTRIBUTED and a.attributed_stage is K and not a.requires_review
    assert [(e.stage, e.status) for e in a.stage_statuses] == [
        (R, DS.NOT_PRESENT), (S, DS.NOT_PRESENT), (P, DS.NOT_PRESENT), (K, DS.FIRST_OBSERVED), (D, DS.PERSISTING)]
    assert a.current_stage_attribution is CA.SUPPORTED and X.DEFECT_PERSISTING in a.flags
    assert r.status is A.ATTRIBUTED and a.missing_stages == []
    # analysed from DISPATCH: the defect is persisting there, dispatch is not blamed
    assert one(run(r.observations, current=D)).current_stage_attribution is CA.NOT_SUPPORTED


def test_case3_pre_existing_not_attributable_to_packing():
    r = run([obs(R, "FAIL", [CRUSHED]), obs(K, "FAIL", [CRUSHED])], current=K)
    a = one(r)
    assert a.classification is C.PRE_EXISTING and a.explanation_code is X.DEFECT_PREEXISTING
    assert a.first_observed_stage is R and a.attribution_status is A.NOT_ATTRIBUTED and a.attributed_stage is None
    assert a.current_stage_status is DS.PERSISTING and a.current_stage_attribution is CA.NOT_SUPPORTED
    assert a.evidence_level is EV.HIGH  # confirmed (FAIL) at receiving


def test_case4_missing_stages_insufficient_evidence():
    r = run([obs(R), obs(K, "FAIL", [CRUSHED])], current=K)
    a = one(r)
    assert a.classification is C.INSUFFICIENT_EVIDENCE and a.explanation_code is X.MISSING_INTERMEDIATE_OBSERVATIONS
    assert a.last_known_good_stage is R and a.first_observed_stage is K and a.missing_stages == [S, P]
    assert a.evidence_level is EV.INSUFFICIENT and a.attribution_status is A.INSUFFICIENT_EVIDENCE
    assert a.attributed_stage is None and a.current_stage_attribution is CA.UNDETERMINED
    assert timeline(r)[S] is TS.NOT_OBSERVED and timeline(r)[P] is TS.NOT_OBSERVED  # never assumed good
    assert r.missing_stages == [S, P] and r.status is A.INSUFFICIENT_EVIDENCE


def test_case5_review_stage_is_not_a_pass():
    r = run([obs(R), obs(P, "REVIEW"), obs(K, "FAIL", [CRUSHED])], current=K)
    a = one(r)
    assert a.classification is C.UNCERTAIN and a.explanation_code is X.PREVIOUS_STAGE_REVIEW
    assert a.requires_review and r.requires_review and r.status is A.REQUIRES_REVIEW
    assert a.previous_inspected_stage is P and a.last_known_good_stage is R  # REVIEW skipped for "good"
    assert timeline(r)[P] is TS.REVIEW and a.evidence_level is EV.LOW


def test_case6_separate_analyses_per_defect_type():
    r = run([obs(K, "FAIL", [CRUSHED]), obs(D, "FAIL", [LEAK])], current=D)
    by = {a.defect_type: a for a in r.defect_analyses}
    assert set(by) == {CRUSHED, LEAK}
    assert by[CRUSHED].classification is C.INSUFFICIENT_EVIDENCE and by[CRUSHED].explanation_code is X.NO_PRIOR_OBSERVATION
    # crushed disappears exactly when leakage appears: possibly the same damage re-classified -> not claimed
    assert by[LEAK].classification is C.UNCERTAIN and by[LEAK].explanation_code is X.AMBIGUOUS_DEFECT_IDENTITY
    assert {e.stage: e.status for e in by[CRUSHED].stage_statuses}[D] is DS.UNCERTAIN


def test_case7_defect_appears_then_resolves():
    r = run([obs(R), obs(S), obs(P, "FAIL", [DENT]), obs(K)], current=P)
    a = one(r)
    assert a.classification is C.NEW_DEFECT and a.first_observed_stage is P and a.evidence_level is EV.HIGH
    assert {e.stage: e.status for e in a.stage_statuses}[K] is DS.RESOLVED
    assert a.final_state is DS.RESOLVED and X.DEFECT_RESOLVED in a.flags


def test_case8_out_of_order_timestamps_become_uncertain():
    # picking photographed BEFORE storage
    r = run([obs(R, minute=0), obs(S, minute=50), obs(P, minute=20), obs(K, "FAIL", [CRUSHED], minute=60)], current=K)
    a = one(r)
    assert X.TIMESTAMP_ORDER_CONFLICT in r.flags
    assert a.classification is C.UNCERTAIN and a.explanation_code is X.TIMESTAMP_ORDER_CONFLICT and a.requires_review
    assert a.evidence_checks["timestamps_consistent"] is False


def test_case9_observations_of_other_items_are_never_combined():
    with pytest.raises(TraceMismatchError):
        run([obs(R), obs(K, "FAIL", [CRUSHED], item="UNIT-2")])
    with pytest.raises(TraceMismatchError):
        run([obs(R), obs(K, sku="SKU-OTHER")])
    order_trace = TraceKey(key_type=TraceKeyType.ORDER, store_id="DS-BLR-001", sku_id="SKU-MILK-500ML", order_id="ORD-1")
    with pytest.raises(TraceMismatchError):
        run([obs(P, order="ORD-1"), obs(K, order="ORD-2")], trace=order_trace)
    assert run([obs(P, order="ORD-1"), obs(K, order="ORD-1")], trace=order_trace).inspected_stages == [P, K]
    with pytest.raises(TraceMismatchError):  # an observation without trace fields is never merged
        run([obs(R), obs(K).model_copy(update={"store_id": None})])


def test_case10_duplicates_are_not_double_counted():
    dup_id = obs(K, "FAIL", [CRUSHED], iid="INS-SAME")
    same_photo = obs(K, "FAIL", [CRUSHED], iid="INS-OTHER", sha="abc", minute=45)
    first_photo = obs(K, "FAIL", [CRUSHED], iid="INS-FIRST", sha="abc", minute=41)
    r = run([obs(R), obs(S), obs(P), dup_id, dup_id, same_photo, first_photo], current=K)
    assert r.duplicates_ignored == 2 and X.DUPLICATE_OBSERVATIONS_IGNORED in r.flags
    k = next(e for e in r.timeline if e.stage is K)
    assert sorted(k.inspection_ids) == ["INS-FIRST", "INS-SAME"] and k.defects == [CRUSHED]
    assert one(r).first_observed_inspection_id == "INS-SAME"  # earliest at the stage (minute 40)
    assert one(r).classification is C.NEW_DEFECT


# ---------------------------------------------------------------- evidence rules and edge cases
def test_multiple_defects_same_stage_are_separate():
    r = run([obs(R), obs(S), obs(P), obs(K, "FAIL", [CRUSHED, LABEL]), obs(D, "FAIL", [CRUSHED, LABEL])])
    assert [a.defect_type for a in r.defect_analyses] == [CRUSHED, LABEL]  # deterministic order
    assert all(a.classification is C.NEW_DEFECT and a.first_observed_stage is K for a in r.defect_analyses)


def test_new_defect_after_defective_stage_is_medium():
    r = run([obs(R), obs(S), obs(P, "FAIL", [LABEL]), obs(K, "FAIL", [LABEL, CRUSHED])], current=K)
    a = next(x for x in r.defect_analyses if x.defect_type is CRUSHED)
    assert a.explanation_code is X.NEW_DEFECT_AFTER_DEFECTIVE_STAGE and a.evidence_level is EV.MEDIUM
    assert a.attribution_status is A.ATTRIBUTED and a.last_known_good_stage is S


def test_unconfirmed_first_observation_lowers_evidence():
    r = run([obs(R), obs(S), obs(P), obs(K, "REVIEW", [CRUSHED])], current=K)
    a = one(r)
    assert a.classification is C.NEW_DEFECT and a.evidence_level is EV.MEDIUM
    assert X.FIRST_OBSERVATION_UNCONFIRMED in a.flags
    r2 = run([obs(R), obs(S), obs(P, "FAIL", [LABEL]), obs(K, "REVIEW", [LABEL, CRUSHED])], current=K)
    low = next(x for x in r2.defect_analyses if x.defect_type is CRUSHED)
    assert low.evidence_level is EV.LOW and low.attribution_status is A.REQUIRES_REVIEW and low.requires_review


def test_operator_feedback_confirms_observation():
    r = run([obs(R), obs(S), obs(P), obs(K, "FAIL", [CRUSHED], source=EvidenceSource.OPERATOR)], current=K)
    assert one(r).evidence_level is EV.HIGH


def test_no_prior_observation():
    a = one(run([obs(K, "FAIL", [CRUSHED])], current=K))
    assert a.classification is C.INSUFFICIENT_EVIDENCE and a.explanation_code is X.NO_PRIOR_OBSERVATION
    assert a.missing_stages == [R, S, P] and a.last_known_good_stage is None


def test_same_stage_pass_and_fail_merge_conservatively():
    r = run([obs(R), obs(S), obs(P), obs(K, iid="INS-A"), obs(K, "FAIL", [CRUSHED], iid="INS-B", minute=41)], current=K)
    assert timeline(r)[K] is TS.FAIL and one(r).classification is C.NEW_DEFECT


def test_empty_and_no_trace():
    r = run([], current=K)
    assert r.status is A.NOT_ATTRIBUTED and r.explanation_code is X.INSUFFICIENT_OBSERVATIONS
    assert all(e.status is TS.NOT_OBSERVED for e in r.timeline)
    none = TraceKey(key_type=TraceKeyType.NONE, store_id="DS-BLR-001", sku_id="SKU-MILK-500ML")
    assert X.NO_TRACE_IDENTIFIER in run([obs(K)], trace=none).flags


def test_deterministic_regardless_of_input_order():
    base = [obs(R), obs(S), obs(P, "REVIEW"), obs(K, "FAIL", [CRUSHED, LABEL]), obs(D, "FAIL", [CRUSHED])]
    expected = run(base, current=K).model_dump()
    rng = random.Random(7)
    for _ in range(20):
        shuffled = base[:]
        rng.shuffle(shuffled)
        assert run(shuffled, current=K).model_dump() == expected


def test_scales_n_log_n():
    many = [obs(FULFILLMENT_FLOW[i % 6], "FAIL" if i % 6 >= 3 else "PASS", [CRUSHED] if i % 6 >= 3 else [],
                minute=i * 10 + (i % 6), iid=f"INS-{i:06d}") for i in range(20_000)]
    t0 = time.perf_counter()
    r = run(many, current=K)
    assert time.perf_counter() - t0 < 5 and len(r.observations) == 20_000
