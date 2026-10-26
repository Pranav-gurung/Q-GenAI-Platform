"""Plain-Python request/response models and the single source of truth for semantic validation.

The FastAPI layer (module_c.api) uses Pydantic models for structure/OpenAPI and calls
``validate_request`` for the rules below, so both layers agree.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

STATUS_SUCCESS, STATUS_INSUFFICIENT, STATUS_ERROR = "success", "insufficient_evidence", "error"
CAUSE_STATUSES = ("hypothesis", "supported_by_document", "requires_investigation")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}$")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MAX_CONTEXT_CHARS = 300


class RequestValidationError(ValueError):
    """Carries a list of {loc, msg} items (rendered as HTTP 422 by the API)."""

    def __init__(self, errors: list):
        super().__init__("; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors))
        self.errors = errors


@dataclass
class Defect:
    defect_class: str
    confidence: float
    bbox: Optional[list] = None  # [x_min, y_min, x_max, y_max], original-image pixels
    mask_ref: Optional[str] = None


@dataclass
class RCARequest:
    inspection_id: str
    image_id: str
    model_version: str
    defects: list
    status: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    part_type: Optional[str] = None
    material: Optional[str] = None
    production_process: Optional[str] = None
    inspection_context: Optional[str] = None


def _clean_text(v, loc, errors, max_len=MAX_CONTEXT_CHARS) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        errors.append({"loc": loc, "msg": "must be a string"})
        return None
    v = _CTRL.sub(" ", v).strip()
    if len(v) > max_len:
        errors.append({"loc": loc, "msg": f"must be at most {max_len} characters"})
        return None
    return v or None


ALLOWED_TOP_LEVEL_KEYS = {
    "inspection_id", "image_id", "model_version", "status", "defects",
    "image_width", "image_height", "part_type", "material",
    "production_process", "inspection_context"
}
ALLOWED_DEFECT_KEYS = {"class", "confidence", "bbox", "mask_ref"}


def validate_request(data: dict, *, supported_classes, max_defects: int) -> RCARequest:
    """Validate a decoded request body (keys as in the API contract; defect class key is ``class``)."""
    errors: list = []
    extra_top = set(data.keys()) - ALLOWED_TOP_LEVEL_KEYS
    for k in sorted(extra_top):
        errors.append({"loc": [k], "msg": "extra fields not permitted"})
    supported = {c.lower() for c in supported_classes}
    for f in ("inspection_id", "image_id", "model_version"):
        v = data.get(f)
        if not isinstance(v, str) or not v.strip():
            errors.append({"loc": [f], "msg": "must be a non-empty string"})
        elif not ID_RE.match(v.strip()):
            errors.append({"loc": [f], "msg": "may contain only letters, digits and . _ : - (max 128 chars)"})
    w, h = data.get("image_width"), data.get("image_height")
    for name, val in (("image_width", w), ("image_height", h)):
        if val is not None and (isinstance(val, bool) or not isinstance(val, int) or val <= 0):
            errors.append({"loc": [name], "msg": "must be a positive integer"})
    if (w is None) != (h is None):
        errors.append({"loc": ["image_width"], "msg": "image_width and image_height must be given together"})
    _ok = lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0  # noqa: E731
    if not (_ok(w) and _ok(h)):
        w = h = None  # unusable dimensions are reported above and never used for bounds checks
    defects_raw = data.get("defects")
    defects: list = []
    if not isinstance(defects_raw, list):
        errors.append({"loc": ["defects"], "msg": "must be a list (it may be empty)"})
        defects_raw = []
    if len(defects_raw) > max_defects:
        errors.append({"loc": ["defects"], "msg": f"at most {max_defects} defects per request (got {len(defects_raw)})"})
        defects_raw = []
    for i, d in enumerate(defects_raw):
        loc = ["defects", i]
        if not isinstance(d, dict):
            errors.append({"loc": loc, "msg": "must be an object"})
            continue
        extra_defect = set(d.keys()) - ALLOWED_DEFECT_KEYS
        for k in sorted(extra_defect):
            errors.append({"loc": loc + [k], "msg": "extra fields not permitted"})
        cls = d.get("class")
        if not isinstance(cls, str) or not cls.strip():
            errors.append({"loc": loc + ["class"], "msg": "must be a non-empty string"})
            cls = None
        elif cls.strip().lower() not in supported:
            errors.append({"loc": loc + ["class"],
                           "msg": f"unsupported defect class {cls.strip()!r}; supported: {sorted(supported)}"})
        conf = d.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            errors.append({"loc": loc + ["confidence"], "msg": "must be a finite number between 0 and 1"})
            conf = None
        bbox = d.get("bbox")
        mask_ref = d.get("mask_ref")
        if bbox is None and mask_ref is None:
            errors.append({"loc": loc + ["bbox"], "msg": "a bbox (or mask_ref) is required"})
        if bbox is not None:
            msg = _bbox_error(bbox, w, h)
            if msg:
                errors.append({"loc": loc + ["bbox"], "msg": msg})
                bbox = None
        if mask_ref is not None and (not isinstance(mask_ref, str) or not mask_ref.strip()):
            errors.append({"loc": loc + ["mask_ref"], "msg": "must be a non-empty string"})
        if cls is not None and conf is not None:
            defects.append(Defect(cls.strip().lower(), float(conf), [float(x) for x in bbox] if bbox else None, mask_ref))
    ctx = {k: _clean_text(data.get(k), [k], errors) for k in ("part_type", "material", "production_process", "inspection_context")}
    status = _clean_text(data.get("status"), ["status"], errors, 64)
    if errors:
        raise RequestValidationError(errors)
    return RCARequest(inspection_id=data["inspection_id"].strip(), image_id=data["image_id"].strip(),
                      model_version=data["model_version"].strip(), defects=defects, status=status,
                      image_width=w, image_height=h, **ctx)


def _bbox_error(bbox, w, h) -> Optional[str]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "must contain exactly 4 numbers [x_min, y_min, x_max, y_max]"
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox):
        return "coordinates must be finite numbers"
    x0, y0, x1, y1 = bbox
    if x0 < 0 or y0 < 0:
        return "coordinates must be non-negative (original-image pixels)"
    if x1 <= x0 or y1 <= y0:
        return "box must have positive width and height (x_max > x_min, y_max > y_min)"
    if w is not None and (x1 > w or y1 > h):
        return f"box exceeds the stated image size {w}x{h}"
    return None


@dataclass
class Passage:
    chunk_id: str
    document_id: str
    title: str
    version: str
    page: int
    section: Optional[str]
    score: float
    text: str
