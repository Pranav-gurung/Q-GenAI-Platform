"""YOLOv8 model loading (Section B).

Design rules:
* The model is loaded ONCE and reused for every request.
* If weights are missing or invalid, the detector reports status
  "unavailable"/"error" and returns NO detections. It never invents results.
* If the weights look like a generic COCO-pretrained model (80 classes incl.
  "person"), it is flagged `is_generic=True` and the inspection rules force a
  REVIEW decision: a COCO model cannot detect scratches, cracks, rust, etc.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.utils.hashing import sha256_file

log = logging.getLogger(__name__)

SUPPORTED_WEIGHT_SUFFIXES = {".pt", ".onnx"}


class ModelUnavailableError(RuntimeError):
    pass


@dataclass
class ModelInfo:
    status: str  # "loaded" | "unavailable" | "error"
    message: str
    name: str | None = None
    version: str | None = None
    sha256: str | None = None
    class_names: list[str] = field(default_factory=list)
    approved: bool = False
    is_generic: bool = False
    backend: str | None = None  # "pytorch" | "onnxruntime"
    device: str | None = None

    def public_dict(self) -> dict:
        return {
            "model_status": self.status,
            "message": self.message,
            "model_name": self.name,
            "model_version": self.version,
            "model_sha256": self.sha256,
            "class_names": self.class_names,
            "approved": self.approved,
            "is_generic_pretrained": self.is_generic,
            "backend": self.backend,
            "device": self.device,
        }


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": round(self.x1, 1), "y1": round(self.y1, 1),
                "x2": round(self.x2, 1), "y2": round(self.y2, 1),
            },
        }


def looks_like_coco(names: list[str]) -> bool:
    return len(names) == 80 and "person" in names and "toothbrush" in names


def resolve_weights_path(models_dir: Path, weights_file: str) -> Path:
    """Resolve a weights FILE NAME inside models_dir; reject path tricks."""
    if not weights_file or Path(weights_file).name != weights_file or weights_file.startswith("."):
        raise ValueError("Weights must be a plain file name inside the models directory.")
    path = (models_dir / weights_file).resolve()
    if path.parent != models_dir.resolve():
        raise ValueError("Weights must be inside the models directory.")
    if path.suffix.lower() not in SUPPORTED_WEIGHT_SUFFIXES:
        raise ValueError("Weights must be a .pt or .onnx file.")
    return path


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model."""

    def __init__(self, models_dir: Path, weights_file: str, version: str,
                 device: str = "cpu", imgsz: int = 640, approved: bool = False):
        self.models_dir = models_dir
        self.weights_file = weights_file
        self.version = version
        self.device = device
        self.imgsz = imgsz
        self.approved = approved
        self._model = None
        self._lock = threading.Lock()  # YOLO predict is not guaranteed thread-safe
        self.info = ModelInfo(status="unavailable", message="Model not loaded yet.",
                              name=weights_file, version=version)

    def load(self) -> ModelInfo:
        try:
            path = resolve_weights_path(self.models_dir, self.weights_file)
        except ValueError as exc:
            self.info = ModelInfo(status="error", message=str(exc), version=self.version)
            return self.info

        if not path.is_file():
            self.info = ModelInfo(
                status="unavailable",
                message=(f"No model weights found ('{self.weights_file}' in the models "
                         "directory). Running in NO-MODEL mode: inspections are routed "
                         "to manual REVIEW and no detections are produced."),
                name=self.weights_file, version=self.version,
            )
            log.warning(self.info.message)
            return self.info

        try:
            from ultralytics import YOLO  # imported lazily: heavy dependency

            model = YOLO(str(path), task="detect")
            names_obj = model.names or {}
            names = [names_obj[k] for k in sorted(names_obj)] if isinstance(names_obj, dict) else list(names_obj)
            checksum = sha256_file(path)
        except Exception as exc:  # corrupt file, wrong format, missing runtime ...
            log.exception("Failed to load model weights")
            self.info = ModelInfo(
                status="error",
                message=f"Model weights could not be loaded ({type(exc).__name__}). "
                        "Inspections are routed to manual REVIEW.",
                name=self.weights_file, version=self.version,
            )
            return self.info

        generic = looks_like_coco(names)
        self._model = model
        self.info = ModelInfo(
            status="loaded",
            message=("WARNING: these weights look like a generic COCO-pretrained model. "
                     "They cannot detect manufacturing defects; every result is routed "
                     "to manual REVIEW." if generic else "Custom defect model loaded."),
            name=self.weights_file, version=self.version, sha256=checksum,
            class_names=names, approved=self.approved, is_generic=generic,
            backend="onnxruntime" if path.suffix.lower() == ".onnx" else "pytorch",
            device=self.device,
        )
        log.info("Loaded model %s (version %s, sha256 %s)", path.name, self.version, checksum[:12])
        return self.info

    def predict(self, rgb: np.ndarray, conf: float) -> list[Detection]:
        if self._model is None or self.info.status != "loaded":
            raise ModelUnavailableError(self.info.message)
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # Ultralytics expects BGR numpy arrays
        with self._lock:
            results = self._model.predict(source=bgr, conf=conf, imgsz=self.imgsz,
                                          device=self.device, verbose=False)
        detections: list[Detection] = []
        names = self.info.class_names
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                detections.append(Detection(
                    class_id=int(k),
                    class_name=names[k] if 0 <= k < len(names) else f"class_{k}",
                    confidence=float(c), x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


class DetectorManager:
    """Holds the single active detector; allows admins to switch models."""

    def __init__(self) -> None:
        self._current: YoloDetector | None = None
        self._lock = threading.Lock()

    @property
    def current(self) -> YoloDetector | None:
        return self._current

    def load(self, models_dir: Path, weights_file: str, version: str, device: str,
             imgsz: int, approved: bool, keep_previous_on_failure: bool = False) -> ModelInfo:
        candidate = YoloDetector(models_dir, weights_file, version, device, imgsz, approved)
        info = candidate.load()
        with self._lock:
            if info.status != "loaded" and keep_previous_on_failure and self._current is not None:
                return info  # keep serving the previously working model
            self._current = candidate
        return info


detector_manager = DetectorManager()


def get_detector():
    """FastAPI dependency. Tests override this to inject a stub detector."""
    return detector_manager.current
