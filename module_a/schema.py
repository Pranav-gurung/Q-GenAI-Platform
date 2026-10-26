"""Canonical internal representation, independent of any export format.

Bounding boxes are ``[x_min, y_min, x_max, y_max]`` in absolute pixel coordinates of the
ORIGINAL image (continuous edge coordinates: x_max may equal image width).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "1.0"
BBOX_FORMAT = "xyxy_abs_pixels"


@dataclass
class Issue:
    code: str
    severity: str  # "error" | "warning"
    message: str
    image_id: Optional[str] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Annotation:
    class_name: str
    bbox: Optional[list] = None  # [x_min, y_min, x_max, y_max]
    mask_ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {"class_name": self.class_name, "bbox": self.bbox, "mask_ref": self.mask_ref}

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(
            class_name=d.get("class_name", d.get("class")),
            bbox=list(d["bbox"]) if d.get("bbox") is not None else None,
            mask_ref=d.get("mask_ref"),
        )


@dataclass
class ImageRecord:
    image_id: str
    file_name: str  # relative to the image root
    width: Optional[int] = None
    height: Optional[int] = None
    group_id: Optional[str] = None  # physical part / capture session / source
    annotations: list = field(default_factory=list)
    sha256: Optional[str] = None
    phash: Optional[str] = None  # 64-bit dHash, hex
    split: Optional[str] = None
    source_image_id: Optional[str] = None  # set only for augmented variants
    aug_ops: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def effective_group(self) -> str:
        return self.group_id or self.image_id

    def to_dict(self) -> dict:
        d = asdict(self)
        d["annotations"] = [a.to_dict() for a in self.annotations]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRecord":
        d = dict(d)
        d["annotations"] = [Annotation.from_dict(a) for a in d.get("annotations", [])]
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def bbox_problem(bbox: Any, width: Optional[int] = None, height: Optional[int] = None) -> Optional[str]:
    """Return a description of what is wrong with ``bbox`` or None if it is valid."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "bbox must be a list of exactly 4 numbers [x_min, y_min, x_max, y_max]"
    for v in bbox:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return "bbox values must be finite numbers"
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return f"bbox has non-positive width/height (x_max<=x_min or y_max<=y_min): {list(bbox)}"
    if x0 < 0 or y0 < 0:
        return f"bbox has negative coordinates: {list(bbox)}"
    if width is not None and x1 > width:
        return f"bbox x_max {x1} exceeds image width {width}"
    if height is not None and y1 > height:
        return f"bbox y_max {y1} exceeds image height {height}"
    return None


def make_class_map(classes: list) -> dict:
    """Stable class-name -> integer id mapping following the configured order."""
    if len(set(classes)) != len(classes):
        raise ValueError("duplicate class names in configuration")
    return {name: i for i, name in enumerate(classes)}


def save_canonical(path: Path, records: list, class_map: dict, meta: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "bbox_format": BBOX_FORMAT,
        "class_map": class_map,
        "meta": meta or {},
        "images": [r.to_dict() for r in records],
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")


def load_canonical(path: Path) -> tuple:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ImageRecord.from_dict(d) for d in doc["images"]], doc.get("class_map", {}), doc.get("meta", {})
