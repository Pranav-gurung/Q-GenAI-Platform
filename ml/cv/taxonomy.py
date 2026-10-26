"""Load and validate the defect taxonomy (ml/cv/taxonomy.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from backend.domain.enums import DefectType, Severity

DATA_STATUSES = ("synthetic_demo", "placeholder", "real")


class TaxonomyError(ValueError):
    pass


@dataclass(frozen=True)
class DefectClass:
    id: int
    name: str
    defect_type: DefectType
    base_severity: Severity
    safety_critical: bool
    data_status: str
    description: str = ""
    min_confidence: float | None = None


@dataclass(frozen=True)
class Taxonomy:
    version: int
    classes: tuple[DefectClass, ...]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    def by_name(self, name: str) -> DefectClass:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(name)

    def by_id(self, class_id: int) -> DefectClass:
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        raise KeyError(class_id)

    def with_status(self, status: str) -> list[DefectClass]:
        return [c for c in self.classes if c.data_status == status]


def parse_taxonomy(data: dict) -> Taxonomy:
    try:
        raw = data["classes"]
        classes = tuple(
            DefectClass(id=int(c["id"]), name=str(c["name"]), defect_type=DefectType(c["defect_type"]),
                        base_severity=Severity(c["base_severity"]), safety_critical=bool(c["safety_critical"]),
                        data_status=str(c["data_status"]), description=str(c.get("description", "")),
                        min_confidence=float(c["min_confidence"]) if c.get("min_confidence") is not None else None)
            for c in raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise TaxonomyError(f"invalid taxonomy entry: {exc}") from exc
    if [c.id for c in classes] != list(range(len(classes))):
        raise TaxonomyError("class ids must be 0..N-1 in order")
    if len({c.name for c in classes}) != len(classes):
        raise TaxonomyError("class names must be unique")
    for c in classes:
        if c.data_status not in DATA_STATUSES:
            raise TaxonomyError(f"{c.name}: data_status must be one of {DATA_STATUSES}")
        if c.min_confidence is not None and not 0 <= c.min_confidence <= 1:
            raise TaxonomyError(f"{c.name}: min_confidence must be in [0, 1]")
    return Taxonomy(version=int(data.get("version", 1)), classes=classes)


@lru_cache
def load_taxonomy(path: Path | str = Path(__file__).with_name("taxonomy.yaml")) -> Taxonomy:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaxonomyError(f"taxonomy file not found: {path}") from exc
    return parse_taxonomy(data)
