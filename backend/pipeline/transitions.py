"""Pipeline state machine: the single definition of legal state transitions.

Happy path:
  CREATED -> IMAGE_RECEIVED -> QUALITY_CHECK -> CV_PROCESSING -> DECISION
          -> PIPELINE_ANALYSIS -> RCA_PROCESSING -> ACTION_GENERATED -> COMPLETED

Justified shortcuts:
  DECISION -> COMPLETED              a confident PASS needs no RCA / corrective action
  <any processing state> -> REQUIRES_REVIEW
                                     a stage cannot decide on its own (image rejected,
                                     low confidence, module not implemented, ...)
  REQUIRES_REVIEW -> COMPLETED       operator resolved it via feedback
  REQUIRES_REVIEW -> IMAGE_RECEIVED  operator retook / re-uploaded the image
  <any non-terminal state> -> FAILED technical failure

COMPLETED and FAILED are terminal.
"""
from __future__ import annotations

from types import MappingProxyType

from backend.core.exceptions import InvalidStateTransitionError
from backend.pipeline.states import PipelineState as S

ALLOWED_TRANSITIONS: MappingProxyType[S, frozenset[S]] = MappingProxyType({
    S.CREATED: frozenset({S.IMAGE_RECEIVED, S.FAILED}),
    S.IMAGE_RECEIVED: frozenset({S.QUALITY_CHECK, S.FAILED}),
    S.QUALITY_CHECK: frozenset({S.CV_PROCESSING, S.REQUIRES_REVIEW, S.FAILED}),
    S.CV_PROCESSING: frozenset({S.DECISION, S.REQUIRES_REVIEW, S.FAILED}),
    S.DECISION: frozenset({S.PIPELINE_ANALYSIS, S.COMPLETED, S.REQUIRES_REVIEW, S.FAILED}),
    S.PIPELINE_ANALYSIS: frozenset({S.RCA_PROCESSING, S.REQUIRES_REVIEW, S.FAILED}),
    S.RCA_PROCESSING: frozenset({S.ACTION_GENERATED, S.REQUIRES_REVIEW, S.FAILED}),
    S.ACTION_GENERATED: frozenset({S.COMPLETED, S.REQUIRES_REVIEW, S.FAILED}),
    S.REQUIRES_REVIEW: frozenset({S.COMPLETED, S.IMAGE_RECEIVED, S.FAILED}),
    S.COMPLETED: frozenset(),
    S.FAILED: frozenset(),
})

assert set(ALLOWED_TRANSITIONS) == set(S), "every state needs a transition entry"


def can_transition(current: S, target: S) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def ensure_transition(current: S, target: S) -> None:
    """Raise InvalidStateTransitionError unless current -> target is legal."""
    if not can_transition(current, target):
        allowed = sorted(s.value for s in ALLOWED_TRANSITIONS[current])
        raise InvalidStateTransitionError(
            f"Illegal pipeline transition {current.value} -> {target.value}",
            details={"from": current.value, "to": target.value, "allowed": allowed},
        )
