import numpy as np
import pytest

from backend.domain.enums import DefectType, Severity
from ml.cv.postprocess import MalformedPredictionError, RawPredictions, aggregate_defects, to_detections
from ml.cv.severity import SeverityPolicy
from ml.cv.taxonomy import load_taxonomy

TAX = load_taxonomy()
CLASS_MAP = {c.id: c for c in TAX.classes}
KW = dict(image_width=400, image_height=200, class_map=CLASS_MAP, confidence_threshold=0.35,
          max_detections=50, severity_policy=SeverityPolicy(0.2))


def raw(boxes, scores, classes):
    return RawPredictions(np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(scores, dtype=np.float32),
                          np.array(classes, dtype=np.float32))


def test_conversion_class_mapping_and_coordinates():
    dets = to_detections(raw([[10, 20, 110, 70]], [0.8], [2]), **KW)
    assert len(dets) == 1
    d = dets[0]
    assert d.defect_type is DefectType.LEAKING_PACKAGE and d.model_class == "leakage"
    assert (d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max) == (10, 20, 110, 70)  # pixels, top-left origin
    assert d.area_ratio == pytest.approx(100 * 50 / (400 * 200), abs=1e-4)
    assert d.severity is Severity.HIGH
    assert d.confidence == pytest.approx(0.8)


def test_confidence_filtering_threshold_inclusive_and_sorted():
    dets = to_detections(raw([[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]], [0.34, 0.35, 0.9], [6, 6, 5]), **KW)
    assert [round(d.confidence, 2) for d in dets] == [0.9, 0.35]


def test_per_class_threshold_override():
    spec = CLASS_MAP[6]
    custom = dict(KW, class_map={**CLASS_MAP, 6: type(spec)(**{**spec.__dict__, "min_confidence": 0.6})})
    dets = to_detections(raw([[0, 0, 10, 10], [0, 0, 10, 10]], [0.5, 0.5], [6, 5]), **custom)
    assert [d.model_class for d in dets] == ["dented_container"]


def test_max_detections_cap():
    n = 10
    dets = to_detections(raw([[0, 0, 10, 10]] * n, np.linspace(0.4, 0.9, n), [1] * n), **dict(KW, max_detections=3))
    assert len(dets) == 3 and dets[0].confidence == pytest.approx(0.9)


def test_small_overshoot_is_clipped():
    d = to_detections(raw([[-1.5, -1, 401.5, 201]], [0.9], [6]), **KW)[0]
    assert (d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max) == (0, 0, 400, 200)
    assert d.area_ratio == 1.0


def test_empty_predictions():
    assert to_detections(RawPredictions.empty(), **KW) == []
    assert aggregate_defects([]) == []


@pytest.mark.parametrize("bad", [
    raw([[0, 0, 10, 10]], [0.9], [42]),                              # unknown class id
    raw([[0, 0, 10, 10]], [1.7], [1]),                               # confidence > 1
    raw([[0, 0, float("nan"), 10]], [0.9], [1]),                     # NaN
    raw([[0, 0, 10, 10]], [0.9], [1.5]),                             # non-integer class
    raw([[0, 0, 500, 10]], [0.9], [1]),                              # far outside the image
    raw([[50, 50, 50, 60]], [0.9], [1]),                             # zero width
    RawPredictions(np.zeros((2, 4)), np.zeros(3), np.zeros(2)),      # inconsistent shapes
    RawPredictions(np.zeros((2, 5)), np.zeros(2), np.zeros(2)),      # wrong box width
])
def test_malformed_model_output_rejected(bad):
    with pytest.raises(MalformedPredictionError):
        to_detections(bad, **KW)


def test_aggregation_per_defect_type():
    dets = to_detections(raw([[0, 0, 10, 10], [0, 0, 300, 150], [0, 0, 20, 20]], [0.5, 0.7, 0.9], [6, 6, 2]), **KW)
    defects = {d.defect_type: d for d in aggregate_defects(dets)}
    box = defects[DefectType.DAMAGED_BOX]
    assert box.detection_count == 2 and box.confidence == pytest.approx(0.7) and box.severity is Severity.MEDIUM
    assert defects[DefectType.LEAKING_PACKAGE].detection_count == 1
