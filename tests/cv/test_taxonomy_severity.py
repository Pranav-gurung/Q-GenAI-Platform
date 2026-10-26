import pytest

from backend.domain.enums import DefectType, Severity
from ml.cv.severity import SeverityPolicy, compute_severity
from ml.cv.taxonomy import TaxonomyError, load_taxonomy, parse_taxonomy

TAX = load_taxonomy()


def test_taxonomy_covers_required_classes_in_order():
    assert TAX.names == ["crushed_packaging", "torn_packaging", "leakage", "broken_seal", "opened_package",
                         "dented_container", "damaged_box", "missing_label", "wrong_item", "visual_anomaly"]
    assert TAX.by_name("leakage").defect_type is DefectType.LEAKING_PACKAGE
    assert TAX.by_name("wrong_item").defect_type is DefectType.INCORRECT_ITEM
    assert {c.name for c in TAX.with_status("synthetic_demo")} == {
        "torn_packaging", "leakage", "dented_container", "damaged_box", "missing_label"}
    assert TAX.with_status("real") == []  # nothing claims real-world data yet


@pytest.mark.parametrize("mutate", [
    lambda c: c[0].update(id=5),                       # ids out of order
    lambda c: c[1].update(name=c[0]["name"]),          # duplicate name
    lambda c: c[0].update(defect_type="NOT_A_TYPE"),   # unknown platform type
    lambda c: c[0].update(base_severity="EXTREME"),
    lambda c: c[0].update(data_status="maybe"),
    lambda c: c[0].update(min_confidence=1.5),
])
def test_invalid_taxonomy_rejected(mutate):
    classes = [{"id": c.id, "name": c.name, "defect_type": c.defect_type.value, "base_severity": c.base_severity.value,
                "safety_critical": c.safety_critical, "data_status": c.data_status} for c in TAX.classes]
    mutate(classes)
    with pytest.raises(TaxonomyError):
        parse_taxonomy({"classes": classes})


POLICY = SeverityPolicy(large_area_ratio=0.2)


@pytest.mark.parametrize("name,area,expected", [
    ("damaged_box", 0.05, Severity.LOW),          # base
    ("damaged_box", 0.20, Severity.MEDIUM),       # large area escalates one level (boundary inclusive)
    ("dented_container", 0.5, Severity.HIGH),
    ("leakage", 0.01, Severity.HIGH),             # safety critical floor
    ("leakage", 0.30, Severity.CRITICAL),         # HIGH + large area
    ("wrong_item", 0.9, Severity.CRITICAL),       # capped at CRITICAL
    ("missing_label", 0.1, Severity.MEDIUM),
])
def test_severity_mapping(name, area, expected):
    assert compute_severity(TAX.by_name(name), area, POLICY) is expected


def test_severity_is_deterministic_and_validated():
    spec = TAX.by_name("torn_packaging")
    assert len({compute_severity(spec, 0.13, POLICY) for _ in range(100)}) == 1
    with pytest.raises(ValueError):
        compute_severity(spec, 1.5, POLICY)
    with pytest.raises(ValueError):
        SeverityPolicy(large_area_ratio=0)
