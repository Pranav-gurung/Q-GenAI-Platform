"""Pipeline states of an inspection."""
from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    CREATED = "CREATED"                      # inspection registered, no image yet
    IMAGE_RECEIVED = "IMAGE_RECEIVED"        # image validated and stored
    QUALITY_CHECK = "QUALITY_CHECK"          # image quality gate running
    CV_PROCESSING = "CV_PROCESSING"          # computer vision running
    DECISION = "DECISION"                    # PASS / FAIL / REVIEW decision running
    PIPELINE_ANALYSIS = "PIPELINE_ANALYSIS"  # where in fulfillment did it happen
    RCA_PROCESSING = "RCA_PROCESSING"        # root-cause analysis running
    ACTION_GENERATED = "ACTION_GENERATED"    # corrective action produced
    COMPLETED = "COMPLETED"                  # terminal: resolved (automatically or by operator)
    FAILED = "FAILED"                        # terminal: technical failure
    REQUIRES_REVIEW = "REQUIRES_REVIEW"      # waiting for a human (or an image retake)


TERMINAL_STATES = frozenset({PipelineState.COMPLETED, PipelineState.FAILED})

# States in which an operator may submit feedback.
FEEDBACK_STATES = frozenset({PipelineState.COMPLETED, PipelineState.REQUIRES_REVIEW})

# States in which an image may be (re)uploaded.
IMAGE_UPLOAD_STATES = frozenset({PipelineState.CREATED, PipelineState.REQUIRES_REVIEW})
