"""Runs the detector and measures real inference time."""
import time
from dataclasses import dataclass

import numpy as np

from app.services.detector import ModelInfo, ModelUnavailableError


@dataclass
class InferenceOutput:
    defects: list[dict]
    inference_time_ms: float | None  # None when no model ran
    model_info: ModelInfo


def run_inference(detector, rgb: np.ndarray, conf: float) -> InferenceOutput:
    if detector is None:
        info = ModelInfo(status="unavailable", message="No detector configured.")
        return InferenceOutput(defects=[], inference_time_ms=None, model_info=info)

    info: ModelInfo = detector.info
    if info.status != "loaded":
        return InferenceOutput(defects=[], inference_time_ms=None, model_info=info)

    start = time.perf_counter()
    try:
        detections = detector.predict(rgb, conf)
    except ModelUnavailableError:
        return InferenceOutput(defects=[], inference_time_ms=None, model_info=detector.info)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return InferenceOutput(
        defects=[d.to_dict() for d in detections],
        inference_time_ms=round(elapsed_ms, 2),
        model_info=info,
    )
