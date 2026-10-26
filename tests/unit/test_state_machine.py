import pytest

from backend.core.exceptions import InvalidStateTransitionError
from backend.pipeline.states import PipelineState as S
from backend.pipeline.transitions import ALLOWED_TRANSITIONS, can_transition, ensure_transition

HAPPY_PATH = [S.CREATED, S.IMAGE_RECEIVED, S.QUALITY_CHECK, S.CV_PROCESSING, S.DECISION, S.PIPELINE_ANALYSIS,
              S.RCA_PROCESSING, S.ACTION_GENERATED, S.COMPLETED]


def test_happy_path_is_valid():
    for current, target in zip(HAPPY_PATH, HAPPY_PATH[1:]):
        ensure_transition(current, target)


@pytest.mark.parametrize("current,target", [
    (S.CREATED, S.COMPLETED),
    (S.CREATED, S.CV_PROCESSING),
    (S.CREATED, S.REQUIRES_REVIEW),
    (S.IMAGE_RECEIVED, S.DECISION),
    (S.QUALITY_CHECK, S.IMAGE_RECEIVED),
    (S.CV_PROCESSING, S.QUALITY_CHECK),
    (S.ACTION_GENERATED, S.CREATED),
    (S.COMPLETED, S.IMAGE_RECEIVED),
    (S.FAILED, S.COMPLETED),
])
def test_invalid_transitions_rejected(current, target):
    assert not can_transition(current, target)
    with pytest.raises(InvalidStateTransitionError) as exc:
        ensure_transition(current, target)
    assert exc.value.code == "INVALID_STATE_TRANSITION"
    assert exc.value.details["from"] == current.value


@pytest.mark.parametrize("current,target", [
    (S.DECISION, S.COMPLETED),            # confident PASS
    (S.QUALITY_CHECK, S.REQUIRES_REVIEW), # image rejected
    (S.REQUIRES_REVIEW, S.COMPLETED),     # operator resolved
    (S.REQUIRES_REVIEW, S.IMAGE_RECEIVED),# retake
    (S.CV_PROCESSING, S.FAILED),
])
def test_justified_shortcuts(current, target):
    assert can_transition(current, target)


def test_terminal_states_have_no_exits():
    assert ALLOWED_TRANSITIONS[S.COMPLETED] == frozenset()
    assert ALLOWED_TRANSITIONS[S.FAILED] == frozenset()


def test_every_non_terminal_state_can_fail():
    for state, targets in ALLOWED_TRANSITIONS.items():
        if state not in (S.COMPLETED, S.FAILED):
            assert S.FAILED in targets, state


def test_transition_table_is_immutable():
    with pytest.raises(TypeError):
        ALLOWED_TRANSITIONS[S.CREATED] = frozenset({S.COMPLETED})
