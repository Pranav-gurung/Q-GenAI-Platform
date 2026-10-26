"""Model artifact loading, device resolution, versioning and in-process caching.

Artifact layout produced by ml/cv/train.py:
    artifacts/cv/<model-version>/
        model/best.pt        weights (never committed to git)
        metadata.json        name, version, framework, classes, training params, data provenance
        metrics.json         validation metrics

Version resolution: CV_MODEL_VERSION (explicit override) > metadata.json "model_version" >
"sha256-<first 12 hex of the weights file>". Every inference reports the resolved version.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.cv.taxonomy import DefectClass, Taxonomy

logger = logging.getLogger(__name__)



class ModelUnavailableError(RuntimeError):
    """The configured model cannot be used. `code` is a stable identifier for logs / health."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    framework: str
    artifact: str            # "<file name>@sha256:<12 hex>" (no absolute paths in API output)
    artifact_sha256: str
    device: str
    train_image_size: int | None
    class_names: tuple[str, ...]
    metadata: dict = field(default_factory=dict, compare=False)


@dataclass
class LoadedModel:
    model: Any                       # ultralytics.YOLO
    info: ModelInfo
    class_map: dict[int, DefectClass]
    lock: threading.Lock = field(default_factory=threading.Lock)  # predictor is not thread-safe


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_metadata(weights: Path) -> dict:
    for candidate in (weights.parent.parent / "metadata.json", weights.parent / "metadata.json"):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("ignoring unreadable model metadata %s", candidate)
    return {}


def resolve_device(requested: str) -> str:
    """'auto' -> cuda:0 if available else cpu. Explicit cuda without a GPU is an error (never faked)."""
    try:
        import torch
    except ImportError as exc:
        raise ModelUnavailableError("CV_DEPENDENCIES_MISSING", "torch is not installed (pip install -e '.[cv]')") from exc
    requested = requested.lower()
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ModelUnavailableError("DEVICE_UNAVAILABLE", f"CV_DEVICE={requested} but CUDA is not available "
                                                              "(set CV_DEVICE=cpu or auto)")
        index = int(requested.split(":")[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            raise ModelUnavailableError("DEVICE_UNAVAILABLE", f"CV_DEVICE={requested}: only "
                                                              f"{torch.cuda.device_count()} CUDA device(s)")
        return f"cuda:{index}"
    raise ModelUnavailableError("INVALID_DEVICE", f"unsupported CV_DEVICE={requested!r}")


def build_class_map(model_names: dict[int, str], taxonomy: Taxonomy) -> dict[int, DefectClass]:
    """Map model class ids to taxonomy classes BY NAME (robust to re-ordered taxonomies)."""
    known = set(taxonomy.names)
    unknown = sorted(set(model_names.values()) - known)
    if unknown:
        raise ModelUnavailableError("TAXONOMY_MISMATCH", f"model classes {unknown} are not in the defect taxonomy")
    return {int(i): taxonomy.by_name(n) for i, n in model_names.items()}


_CACHE: dict[tuple, LoadedModel] = {}
_CACHE_LOCK = threading.Lock()


def load_model(weights: Path, device: str, taxonomy: Taxonomy, version_override: str | None = None) -> LoadedModel:
    """Load (or return the cached) model. Raises ModelUnavailableError with a specific code."""
    weights = Path(weights)
    if not weights.is_file():
        raise ModelUnavailableError("MODEL_NOT_FOUND", f"model artifact not found: {weights} "
                                                       "(train one with `python -m ml.cv.train`, see README)")
    resolved_device = resolve_device(device)
    stat = weights.stat()
    key = (str(weights.resolve()), stat.st_mtime_ns, stat.st_size, resolved_device, version_override, id(taxonomy))
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
        try:
            from ml.cv._ultralytics import import_ultralytics
            ultralytics = import_ultralytics()
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailableError("CV_DEPENDENCIES_MISSING",
                                        "ultralytics/torch not installed (pip install -e '.[cv]')") from exc
        try:
            model = YOLO(str(weights), task="detect")
            if model.task != "detect":
                raise ModelUnavailableError("MODEL_LOAD_FAILED", f"artifact is a '{model.task}' model, expected 'detect'")
            model.to(resolved_device)
        except ModelUnavailableError:
            raise
        except Exception as exc:  # corrupt file, incompatible checkpoint, ...
            raise ModelUnavailableError("MODEL_LOAD_FAILED", f"cannot load model artifact {weights.name}: "
                                                             f"{type(exc).__name__}") from exc
        names = {int(k): str(v) for k, v in model.names.items()}
        class_map = build_class_map(names, taxonomy)
        meta = find_metadata(weights)
        digest = _sha256(weights)
        version = version_override or meta.get("model_version") or f"sha256-{digest[:12]}"
        train_args = (getattr(model, "ckpt", None) or {}).get("train_args") or {}
        imgsz = train_args.get("imgsz")
        info = ModelInfo(
            name=meta.get("model_name") or Path(str(train_args.get("model") or weights.stem)).stem,
            version=str(version),
            framework=f"ultralytics {ultralytics.__version__} / torch {torch.__version__}",
            artifact=f"{weights.name}@sha256:{digest[:12]}", artifact_sha256=digest, device=resolved_device,
            train_image_size=int(imgsz) if isinstance(imgsz, int) else None,
            class_names=tuple(names[i] for i in sorted(names)), metadata=meta)
        loaded = LoadedModel(model=model, info=info, class_map=class_map)
        _CACHE[key] = loaded
        logger.info("CV model loaded: %s version=%s device=%s artifact=%s", info.name, info.version,
                    info.device, info.artifact)
        return loaded


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
