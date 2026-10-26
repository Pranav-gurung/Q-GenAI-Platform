"""Read annotation metadata (CSV or JSON) into canonical ImageRecords.

Nothing is silently skipped: rows that cannot be parsed produce Issues and the affected rows are
excluded from the returned records (and reported), so the caller decides whether to proceed.

CSV columns (case-insensitive; aliases in parentheses):
  image (file_name, filename)            required, path relative to the image root
  class (class_name, label)              empty for defect-free images
  xmin,ymin,xmax,ymax (x_min,...)        empty for defect-free images
  image_id, width, height, group_id      optional
JSON: either the canonical file written by this module, or a list / {"images": [...]} of
  {"file_name"|"image", "image_id"?, "width"?, "height"?, "group_id"?,
   "annotations": [{"class"|"class_name", "bbox": [x0,y0,x1,y1]}]}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import Annotation, ImageRecord, Issue

_ALIASES = {
    "image": ("image", "file_name", "filename", "file"),
    "class": ("class", "class_name", "label"),
    "x0": ("xmin", "x_min"),
    "y0": ("ymin", "y_min"),
    "x1": ("xmax", "x_max"),
    "y1": ("ymax", "y_max"),
    "image_id": ("image_id",),
    "width": ("width",),
    "height": ("height",),
    "group_id": ("group_id", "group", "part_id", "session_id"),
}


def default_image_id(file_name: str) -> str:
    p = Path(file_name).with_suffix("")
    return "__".join(p.parts)


def _to_int(v):
    return int(float(v)) if v not in (None, "") else None


def load_csv(path: Path) -> tuple:
    issues: list = []
    records: dict = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        pick = {k: next((cols[a] for a in al if a in cols), None) for k, al in _ALIASES.items()}
        if pick["image"] is None:
            return [], [Issue("csv_missing_column", "error", "CSV needs an 'image' (or file_name) column")]
        for lineno, row in enumerate(reader, start=2):
            fname = (row.get(pick["image"]) or "").strip()
            if not fname:
                issues.append(Issue("csv_row_no_image", "error", f"line {lineno}: empty image name"))
                continue
            iid = (row.get(pick["image_id"]) or "").strip() if pick["image_id"] else ""
            iid = iid or default_image_id(fname)
            try:
                rec = records.get(iid) or ImageRecord(
                    image_id=iid, file_name=fname,
                    width=_to_int(row.get(pick["width"])) if pick["width"] else None,
                    height=_to_int(row.get(pick["height"])) if pick["height"] else None,
                    group_id=((row.get(pick["group_id"]) or "").strip() or None) if pick["group_id"] else None)
            except ValueError:
                issues.append(Issue("csv_bad_dimension", "error", f"line {lineno}: non-numeric width/height", iid))
                continue
            if rec.file_name != fname:
                issues.append(Issue("duplicate_image_id", "error",
                                    f"line {lineno}: image_id {iid!r} used for different files", iid))
                continue
            records[iid] = rec
            cls_name = (row.get(pick["class"]) or "").strip() if pick["class"] else ""
            coords = [(row.get(pick[k]) if pick[k] else None) for k in ("x0", "y0", "x1", "y1")]
            if not cls_name and all(c in (None, "") for c in coords):
                continue  # defect-free image row
            try:
                bbox = [float(c) for c in coords]
            except (TypeError, ValueError):
                issues.append(Issue("annotation_unparsable", "error",
                                    f"line {lineno}: bbox values missing or non-numeric: {coords}", iid))
                continue
            rec.annotations.append(Annotation(class_name=cls_name, bbox=bbox))
    return list(records.values()), issues


def load_json(path: Path) -> tuple:
    issues: list = []
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [], [Issue("json_invalid", "error", f"annotation file is not valid JSON: {e}")]
    items = doc.get("images") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        return [], [Issue("json_structure", "error", "expected a list or an object with an 'images' list")]
    records, seen = [], set()
    for n, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(Issue("json_structure", "error", f"entry {n} is not an object"))
            continue
        fname = item.get("file_name") or item.get("image")
        if not fname:
            issues.append(Issue("json_missing_field", "error", f"entry {n}: missing file_name/image"))
            continue
        iid = item.get("image_id") or default_image_id(fname)
        if iid in seen:
            issues.append(Issue("duplicate_image_id", "error", f"duplicate image_id {iid!r}", iid))
            continue
        seen.add(iid)
        anns, bad = [], False
        for a in item.get("annotations") or []:
            name = a.get("class_name", a.get("class")) if isinstance(a, dict) else None
            if not name or not isinstance(a, dict):
                issues.append(Issue("annotation_structure", "error", "annotation without class name", iid))
                bad = True
                continue
            anns.append(Annotation(class_name=name,
                                   bbox=list(a["bbox"]) if a.get("bbox") is not None else None,
                                   mask_ref=a.get("mask_ref")))
        if bad:
            continue
        records.append(ImageRecord(image_id=iid, file_name=fname, width=item.get("width"),
                                   height=item.get("height"), group_id=item.get("group_id"), annotations=anns))
    return records, issues


def load_annotations(path: Path) -> tuple:
    path = Path(path)
    if not path.exists():
        return [], [Issue("annotation_file_missing", "error", f"annotation file not found: {path}")]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".json":
        return load_json(path)
    return [], [Issue("annotation_format", "error", f"unsupported annotation file type {suffix!r} (use .csv or .json)")]
