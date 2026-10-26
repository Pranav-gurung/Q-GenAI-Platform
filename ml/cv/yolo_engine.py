"""YoloCVEngine: the real CVEngine (CV_ENGINE=yolov8), backed by Ultralytics YOLO.

    ImageInput -> load bytes -> decode + EXIF orientation (preprocessing.py)
               -> YOLO predict (letterbox to image size, NMS)       [real inference]
               -> postprocess.py (validate, class map, per-class threshold, clip, severity)
               -> CVResult (detections, aggregated defects, inference metadata)

Guarantees:
  * If the model cannot be loaded (missing/corrupt artifact, CUDA requested but absent, deps missing)
    the engine stays constructible, reports `available: False` in /api/health, and every detect()
    returns status=UNAVAILABLE with NO detections. It never falls back to fake predictions.
  * Errors during inference / malformed model output raise, and the orchestrator marks the
    inspection FAILED (MODULE_ERROR) without leaking details to API clients.
  * The engine never touches the database or API and never decides PASS/FAIL.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from backend.core.config import Settings
from backend.domain.enums import ModuleStatus
from backend.pipeline.context import ImageInput
from backend.pipeline.interfaces import CVEngine
from backend.schemas.results import CVInferenceMetadata, CVResult
from ml.cv.model import LoadedModel, ModelUnavailableError, load_model
from ml.cv.postprocess import RawPredictions, aggregate_defects, to_detections
from ml.cv.preprocessing import to_model_input
from ml.cv.severity import SeverityPolicy
from ml.cv.taxonomy import TaxonomyError, load_taxonomy

logger = logging.getLogger(__name__)
DEFAULT_IMAGE_SIZE = 640


class YoloCVEngine(CVEngine):
    name = "yolov8"

    def __init__(self, settings: Settings, *, warmup: bool = True) -> None:
        self.settings = settings
        self.loaded: LoadedModel | None = None
        self.error: ModelUnavailableError | None = None
        self.severity_policy = SeverityPolicy(settings.cv_severity_large_area_ratio)
        try:
            taxonomy = load_taxonomy(settings.cv_taxonomy_path)
            self.loaded = load_model(settings.cv_model_path, settings.cv_device, taxonomy, settings.cv_model_version)
            if warmup:
                self._warmup()
        except TaxonomyError as exc:
            self.error = ModelUnavailableError("TAXONOMY_INVALID", str(exc))
        except ModelUnavailableError as exc:
            self.error = exc
        except Exception as exc:  # e.g. warm-up inference crashed on this device
            logger.exception("CV model warm-up failed")
            self.error = ModelUnavailableError("MODEL_LOAD_FAILED", f"model warm-up failed: {type(exc).__name__}")
        if self.error:
            self.loaded = None
            logger.error("CV engine unavailable: %s: %s", self.error.code, self.error.message)

    # ------------------------------------------------------------------ properties
    @property
    def available(self) -> bool:
        return self.loaded is not None

    @property
    def image_size(self) -> int:
        if self.settings.cv_image_size:
            return self.settings.cv_image_size
        if self.loaded and self.loaded.info.train_image_size:
            return self.loaded.info.train_image_size
        return DEFAULT_IMAGE_SIZE

    @property
    def predict_conf(self) -> float:
        """Lowest threshold in use; per-class thresholds are applied in postprocessing."""
        per_class = [c.min_confidence for c in self.loaded.class_map.values() if c.min_confidence is not None]
        return min([self.settings.cv_confidence_threshold, *per_class])

    def health(self) -> dict:
        if not self.loaded:
            return {"available": False, "model_loaded": False, "model_version": None, "device": None,
                    "error": f"{self.error.code}: {self.error.message}" if self.error else "not loaded"}
        info = self.loaded.info
        return {"available": True, "model_loaded": True, "model_name": info.name, "model_version": info.version,
                "device": info.device, "framework": info.framework, "artifact": info.artifact,
                "image_size": self.image_size, "confidence_threshold": self.settings.cv_confidence_threshold,
                "classes": list(info.class_names),
                "data_provenance": info.metadata.get("data_provenance", "unknown")}

    # ------------------------------------------------------------------ inference
    def _warmup(self) -> None:
        dummy = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        self._predict(dummy)

    def _predict(self, bgr: np.ndarray):
        s = self.settings
        with self.loaded.lock:
            return self.loaded.model.predict(
                source=bgr, imgsz=self.image_size, conf=self.predict_conf, iou=s.cv_iou_threshold,
                max_det=s.cv_max_detections, device=self.loaded.info.device, augment=False, verbose=False)

    def detect(self, image: ImageInput) -> CVResult:
        if not self.loaded:
            code = self.error.code if self.error else "MODEL_NOT_LOADED"
            msg = self.error.message if self.error else "model not loaded"
            return CVResult.unavailable(self.name, f"{code}: {msg}"[:500])

        t0 = time.perf_counter()
        bgr, width, height = to_model_input(image.load_bytes())
        decode_ms = (time.perf_counter() - t0) * 1000
        results = self._predict(bgr)
        if len(results) != 1:
            raise ValueError(f"expected 1 result from the model, got {len(results)}")
        r = results[0]
        boxes = r.boxes
        raw = RawPredictions.empty() if boxes is None else RawPredictions(
            boxes_xyxy=boxes.xyxy.cpu().numpy(), scores=boxes.conf.cpu().numpy(),
            class_ids=boxes.cls.cpu().numpy())

        detections = to_detections(raw, image_width=width, image_height=height,
                                   class_map=self.loaded.class_map,
                                   confidence_threshold=self.settings.cv_confidence_threshold,
                                   max_detections=self.settings.cv_max_detections,
                                   severity_policy=self.severity_policy)
        speed = getattr(r, "speed", {}) or {}
        info = self.loaded.info
        metadata = CVInferenceMetadata(
            model_name=info.name, model_version=info.version, model_framework=info.framework,
            model_artifact=info.artifact, inference_device=info.device, image_size=self.image_size,
            image_width=width, image_height=height,
            confidence_threshold=self.settings.cv_confidence_threshold, iou_threshold=self.settings.cv_iou_threshold,
            max_detections=self.settings.cv_max_detections,
            preprocess_ms=round(decode_ms + float(speed.get("preprocess") or 0), 2),
            inference_time_ms=round(float(speed.get("inference") or 0), 2),
            postprocess_ms=round(float(speed.get("postprocess") or 0), 2),
            raw_detection_count=len(raw))
        return CVResult(
            module=self.name, module_version=info.version, status=ModuleStatus.COMPLETED,
            detections=detections, defects=aggregate_defects(detections),
            classification="DEFECTS_DETECTED" if detections else "NO_DEFECTS_DETECTED",
            overall_confidence=max((d.confidence for d in detections), default=None),
            inference=metadata,
            message=f"{len(detections)} detection(s) by {info.name}:{info.version} on {info.device}",
            processing_time_ms=round((time.perf_counter() - t0) * 1000, 2))
