"""Validate a YOLO-format detection dataset (e.g. the Section A output).

Usage:
    python -m training.validate_dataset --data dataset/data.yaml

Checks:
  * data.yaml exists and defines `names` (list or {id: name}) and train/val paths
  * `nc`, if present, matches the number of names
  * each split's images/ dir exists and contains .jpg/.jpeg/.png files
  * every label line is "class cx cy w h" with a valid class id and coords in [0, 1]
  * reports images without labels (background images - allowed, but counted)
    and label files without images (orphans - error)
Exit code 0 = usable, 1 = errors found.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_names(cfg: dict) -> list[str]:
    names = cfg.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError("data.yaml must define 'names' as a list or a {id: name} mapping.")


def _images_to_labels(img_dir: Path) -> Path:
    # YOLO convention: .../images/<split>  ->  .../labels/<split>
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels" / img_dir.name


def validate_dataset(data_yaml: Path, max_errors: int = 50) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    report: dict = {"data_yaml": str(data_yaml), "splits": {}, "errors": errors, "warnings": warnings,
                    "class_names": [], "valid": False}

    if not data_yaml.is_file():
        errors.append(f"{data_yaml} not found.")
        return report
    try:
        cfg = yaml.safe_load(data_yaml.read_text()) or {}
        names = load_names(cfg)
    except (yaml.YAMLError, ValueError) as exc:
        errors.append(f"Invalid data.yaml: {exc}")
        return report
    report["class_names"] = names
    if "nc" in cfg and int(cfg["nc"]) != len(names):
        errors.append(f"nc={cfg['nc']} but {len(names)} names are defined.")
    if not names:
        errors.append("No class names defined.")

    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    for split in ("train", "val", "test"):
        if split not in cfg or cfg[split] in (None, ""):
            if split in ("train", "val"):
                errors.append(f"data.yaml has no '{split}' entry.")
            continue
        img_dir = Path(cfg[split])
        img_dir = img_dir if img_dir.is_absolute() else root / img_dir
        lbl_dir = _images_to_labels(img_dir)
        info = {"images": 0, "labelled_images": 0, "background_images": 0, "instances": 0,
                "per_class": Counter()}
        report["splits"][split] = info
        if not img_dir.is_dir():
            errors.append(f"[{split}] image directory missing: {img_dir}")
            continue
        images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
        info["images"] = len(images)
        if not images:
            errors.append(f"[{split}] no images in {img_dir}")
        stems = set()
        for img in images:
            rel = img.relative_to(img_dir).with_suffix(".txt")
            stems.add(str(rel))
            lbl = lbl_dir / rel
            if not lbl.is_file():
                info["background_images"] += 1
                continue
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            if not lines:
                info["background_images"] += 1
                continue
            info["labelled_images"] += 1
            for n, line in enumerate(lines, 1):
                parts = line.split()
                where = f"[{split}] {lbl.name}:{n}"
                if len(parts) != 5:
                    errors.append(f"{where} expected 5 values, got {len(parts)} "
                                  "(segmentation labels are not supported here)")
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    errors.append(f"{where} non-numeric value")
                    continue
                if not 0 <= cls < len(names):
                    errors.append(f"{where} class id {cls} out of range 0..{len(names) - 1}")
                    continue
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                    errors.append(f"{where} coordinates must be normalised to [0, 1] with w, h > 0")
                    continue
                info["instances"] += 1
                info["per_class"][names[cls]] += 1
        if lbl_dir.is_dir():
            orphans = [p for p in lbl_dir.rglob("*.txt") if str(p.relative_to(lbl_dir)) not in stems]
            if orphans:
                errors.append(f"[{split}] {len(orphans)} label file(s) without an image, e.g. {orphans[0].name}")
        else:
            warnings.append(f"[{split}] label directory missing: {lbl_dir} (all images are background)")
        if info["images"] and info["background_images"] == info["images"]:
            warnings.append(f"[{split}] contains no labelled objects.")
        info["per_class"] = dict(info["per_class"])

    train_classes = set(report["splits"].get("train", {}).get("per_class", {}))
    missing = [n for n in names if n not in train_classes]
    if names and "train" in report["splits"] and missing:
        warnings.append(f"Classes with no training instances: {missing}")

    if len(errors) > max_errors:
        extra = len(errors) - max_errors
        del errors[max_errors:]
        errors.append(f"... and {extra} more errors")
    report["valid"] = not errors
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/data.yaml", type=Path)
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args()
    rep = validate_dataset(args.data)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"Dataset: {rep['data_yaml']}")
        print(f"Classes: {rep.get('class_names')}")
        for split, s in rep["splits"].items():
            print(f"  {split:5s}: {s['images']} images, {s['labelled_images']} labelled, "
                  f"{s['background_images']} background, {s['instances']} boxes {s['per_class']}")
        for w in rep["warnings"]:
            print(f"WARNING: {w}")
        for e in rep["errors"]:
            print(f"ERROR: {e}")
        print("RESULT:", "VALID" if rep["valid"] else "INVALID")
    return 0 if rep["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
