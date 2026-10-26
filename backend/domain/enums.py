"""Enumerations for inspection results. Values are stable API/DB strings."""
from __future__ import annotations

from enum import Enum


class ModuleStatus(str, Enum):
    """Outcome of one pipeline module run.

    COMPLETED        the module produced a real result
    NOT_IMPLEMENTED  placeholder module; produced NO result (never fake values)
    SKIPPED          module intentionally not run (e.g. required upstream input missing)
    FAILED           module raised an error
    """
    COMPLETED = "COMPLETED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class DecisionOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class DecisionSource(str, Enum):
    MODEL = "MODEL"      # derived from CV model output
    RULES = "RULES"      # deterministic business rules
    HUMAN = "HUMAN"      # operator decision


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DefectType(str, Enum):
    """Initial defect taxonomy for dark-store products. Part 2 aligns the CV class map with this."""
    CRUSHED_PACKAGE = "CRUSHED_PACKAGE"
    TORN_PACKAGE = "TORN_PACKAGE"
    LEAKING_PACKAGE = "LEAKING_PACKAGE"
    BROKEN_SEAL = "BROKEN_SEAL"
    OPENED_PACKAGE = "OPENED_PACKAGE"
    DENTED_CONTAINER = "DENTED_CONTAINER"
    DAMAGED_BOX = "DAMAGED_BOX"
    INCORRECT_ITEM = "INCORRECT_ITEM"
    MISSING_LABEL = "MISSING_LABEL"
    VISUAL_ANOMALY = "VISUAL_ANOMALY"


class ActionPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class LearningStatus(str, Enum):
    """Lifecycle of a feedback record in the (future) active-learning loop."""
    PENDING = "PENDING"        # stored, not yet reviewed for training
    ACCEPTED = "ACCEPTED"      # approved as a training label
    REJECTED = "REJECTED"      # excluded from training
    USED = "USED"              # included in a training dataset version
