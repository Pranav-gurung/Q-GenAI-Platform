"""Dataset validation: files, image readability, dimensions, classes, boxes, duplicates.

Usage:
  python -m module_a.validate_dataset --image-dir D --annotations A.csv --classes a,b --report out.json
Exit code 1 if any error-level issue exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image

from .hashing import dhash, hamming_hex, sha256_file
from .ingest import load_annotations
from .schema import ImageRecord, Issue, bbox_problem


class ValidationReport:
    def __init__(self, issues: list, n_images: int, invalid_image_ids: set, params: dict):
        self.issues, self.n_images, self.invalid_image_ids, self.params = issues, n_images, invalid_image_ids, params

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        counts: dict = defaultdict(int)
        for i in self.issues:
            counts[f"{i.severity}:{i.code}"] += 1
        return {"valid": self.valid, "n_images_checked": self.n_images,
                "n_errors": len(self.errors), "n_warnings": len(self.warnings),
                "issue_counts": dict(sorted(counts.items())),
                "invalid_image_ids": sorted(self.invalid_image_ids),
                "params": self.params, "issues": [i.to_dict() for i in self.issues]}

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def validate_records(records: list, image_root: Path, classes: list, *, allow_empty: bool = True,
                     near_dup_threshold: int = 4, parse_issues: Optional[list] = None) -> ValidationReport:
    """Validate and enrich records in place (sha256, phash, width, height are filled in)."""
    issues = list(parse_issues or [])
    class_set = set(classes)
    bad: set = set()

    def add(code, sev, msg, rec=None, **details):
        issues.append(Issue(code, sev, msg, rec.image_id if rec else None, details))
        if sev == "error" and rec is not None:
            bad.add(rec.image_id)

    for rec in records:
        path = Path(image_root) / rec.file_name
        if not path.is_file():
            add("missing_image", "error", f"image file not found: {path}", rec)
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                actual_w, actual_h = im.size
                rec.phash = dhash(im)
        except Exception as e:  # corrupted / unsupported
            add("unreadable_image", "error", f"image cannot be decoded: {type(e).__name__}: {e}", rec)
            continue
        rec.sha256 = sha256_file(path)
        if (rec.width, rec.height) != (None, None) and (rec.width, rec.height) != (actual_w, actual_h):
            add("dimension_mismatch", "error",
                f"declared size {rec.width}x{rec.height} != actual {actual_w}x{actual_h}", rec)
        rec.width, rec.height = actual_w, actual_h
        if not rec.annotations:
            add("empty_annotation", "error" if not allow_empty else "warning",
                "image has no annotations (treated as defect-free)", rec)
        for a in rec.annotations:
            if a.class_name not in class_set:
                add("unknown_class", "error", f"class {a.class_name!r} is not in the configured classes", rec,
                    configured=sorted(class_set))
            if a.mask_ref is not None and a.bbox is None:
                continue  # mask-only annotation: mask content is validated by the mask adapter (not implemented)
            problem = bbox_problem(a.bbox, actual_w, actual_h)
            if problem:
                code = "bbox_out_of_bounds" if "exceeds" in problem or "negative" in problem else "invalid_bbox"
                add(code, "error", problem, rec, bbox=a.bbox)

    # duplicates (reported, never silently dropped)
    by_hash: dict = defaultdict(list)
    for r in records:
        if r.sha256:
            by_hash[r.sha256].append(r)
    for h, group in by_hash.items():
        if len(group) > 1:
            issues.append(Issue("exact_duplicate", "warning",
                                f"{len(group)} files have identical content", None,
                                {"image_ids": [g.image_id for g in group], "sha256": h}))
    # one representative per distinct file content (exact duplicates are already reported above)
    hashed = [grp[0] for grp in by_hash.values() if grp[0].phash]
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            d = hamming_hex(hashed[i].phash, hashed[j].phash)
            if d <= near_dup_threshold:
                issues.append(Issue("near_duplicate", "warning",
                                    f"perceptual hash distance {d} (<= {near_dup_threshold}); possible near-duplicates",
                                    None, {"image_ids": [hashed[i].image_id, hashed[j].image_id], "distance": d}))
    return ValidationReport(issues, len(records), bad,
                            {"allow_empty": allow_empty, "near_dup_threshold": near_dup_threshold,
                             "classes": list(classes)})


def parse_classes(value: str) -> list:
    p = Path(value)
    if p.is_file():
        txt = p.read_text(encoding="utf-8")
        try:
            data = json.loads(txt)
            return list(data) if isinstance(data, list) else list(data.keys())
        except json.JSONDecodeError:
            return [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return [c.strip() for c in value.split(",") if c.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--classes", required=True, help="comma list, or path to a text/JSON file")
    ap.add_argument("--report", default="data/reports/validation.json")
    ap.add_argument("--no-empty", action="store_true", help="treat images without annotations as errors")
    ap.add_argument("--near-dup-threshold", type=int, default=4)
    a = ap.parse_args(argv)
    records, parse_issues = load_annotations(Path(a.annotations))
    rep = validate_records(records, Path(a.image_dir), parse_classes(a.classes), allow_empty=not a.no_empty,
                           near_dup_threshold=a.near_dup_threshold, parse_issues=parse_issues)
    rep.save(Path(a.report))
    d = rep.to_dict()
    print(f"images={d['n_images_checked']} errors={d['n_errors']} warnings={d['n_warnings']} -> {a.report}")
    return 0 if rep.valid else 1


if __name__ == "__main__":
    sys.exit(main())
