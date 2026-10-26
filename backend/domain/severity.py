"""Severity ordering helpers (shared by CV severity mapping and decision rules)."""
from __future__ import annotations

from backend.domain.enums import Severity

SEVERITY_ORDER: tuple[Severity, ...] = (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)


def severity_rank(severity: Severity) -> int:
    return SEVERITY_ORDER.index(severity)


def max_severity(values) -> Severity | None:
    values = [v for v in values if v is not None]
    return max(values, key=severity_rank) if values else None


def escalate(severity: Severity, steps: int = 1) -> Severity:
    return SEVERITY_ORDER[min(severity_rank(severity) + steps, len(SEVERITY_ORDER) - 1)]
