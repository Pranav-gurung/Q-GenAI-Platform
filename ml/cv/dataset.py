"""Object-detection dataset (YOLO format) validation.

Layout (one directory per dataset version, e.g. data/cv/synthetic-v1/):
    dataset.yaml                    Ultralytics data file (train/val/test dirs + class names)
    images/{train,val,test}/*.jpg|png|webp
    labels/{train,val,test}/*.txt   one line per box: "<class_id> <xc> <yc> <w> <h>" normalised 0-1;
                                    an EMPTY file means "no defects" (explicit negative example)

Errors (dataset unusable): missing dirs/yaml, class names not matching the taxonomy, unsupported or
corrupt images, image without label file, label file without image, malformed lines, invalid class
ids, non-finite / out-of-range coordinates, boxes outside the image, zero-area boxes.
Warnings (suspicious): exact duplicate boxes, near-duplicate same-class boxes (IoU > 0.95),
boxes smaller than 2 px, identical images in more than one split (train/test leakage),
classes without any training instances.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

from ml.cv.taxonomy import Taxonomy

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")
_EPS = 1e-6


@dataclass
class Issue:
    level: str  # error | warning
    code: str
    path: str
    message: str


@dataclass
class DatasetReport:
    dataset: str
    issues: list[Issue] = field(default_factory=list)
    images: dict[str, int] = field(default_factory=dict)
    instances: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> class -> count
    negatives: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "ok": self.ok, "error_count": len(self.errors),
                "warning_count": len(self.warnings), "images": self.images, "negatives": self.negatives,
                "instances": self.instances, "issues": [asdict(i) for i in self.issues]}


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dataset_root(data_yaml: Path, cfg: dict) -> Path:
    """Directory the split paths are relative to: `path:` (relative to the yaml) or the yaml's folder."""
    base = Path(data_yaml).resolve().parent
    root = Path(cfg["path"]) if cfg.get("path") else base
    return root if root.is_absolute() else (base / root).resolve()


def validate_dataset(data_yaml: Path, taxonomy: Taxonomy) -> DatasetReport:
    data_yaml = Path(data_yaml)
    report = DatasetReport(dataset=str(data_yaml))

    def add(level, code, path, msg):
        report.issues.append(Issue(level, code, str(path), msg))

    if not data_yaml.is_file():
        add("error", "MISSING_DATASET_YAML", data_yaml, "dataset.yaml not found")
        return report
    try:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        add("error", "INVALID_DATASET_YAML", data_yaml, f"cannot parse: {exc}")
        return report
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names != taxonomy.names:
        add("error", "CLASS_NAMES_MISMATCH", data_yaml,
            f"names {names} do not match taxonomy {taxonomy.names} (regenerate dataset.yaml from the taxonomy)")
        return report
    root = dataset_root(data_yaml, cfg)

    hashes: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        if split not in cfg:
            if split != "test":
                add("error", "MISSING_SPLIT", data_yaml, f"'{split}' split not declared")
            continue
        img_dir = root / cfg[split]
        lbl_dir = root / str(cfg[split]).replace("images", "labels", 1)
        if not img_dir.is_dir():
            add("error", "MISSING_IMAGE_DIR", img_dir, "image directory not found")
            continue
        if not lbl_dir.is_dir():
            add("error", "MISSING_LABEL_DIR", lbl_dir, "label directory not found")
            continue
        counts: Counter = Counter()
        n_images = negatives = 0
        image_stems = set()
        for img_path in sorted(p for p in img_dir.iterdir() if p.is_file()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                add("error", "UNSUPPORTED_IMAGE_FORMAT", img_path, f"extension {img_path.suffix!r} not supported")
                continue
            image_stems.add(img_path.stem)
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    im.load()
                    width, height = im.size
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
                add("error", "CORRUPT_IMAGE", img_path, "cannot decode image")
                continue
            n_images += 1
            hashes[hashlib.sha256(img_path.read_bytes()).hexdigest()].add(split)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                add("error", "MISSING_LABEL_FILE", img_path,
                    "no label file (use an empty .txt for images without defects)")
                continue
            lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                negatives += 1
            seen_lines: set[str] = set()
            boxes: list[tuple[int, tuple[float, float, float, float]]] = []
            for n, line in enumerate(lines, 1):
                where = f"{lbl_path}:{n}"
                parts = line.split()
                if len(parts) != 5:
                    add("error", "MALFORMED_LABEL", where, f"expected 5 values, got {len(parts)}")
                    continue
                try:
                    cls_f = float(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    add("error", "MALFORMED_LABEL", where, "non-numeric value")
                    continue
                if not cls_f.is_integer() or not 0 <= int(cls_f) < len(taxonomy.classes):
                    add("error", "INVALID_CLASS_ID", where, f"class id {parts[0]} not in 0..{len(taxonomy.classes) - 1}")
                    continue
                if not all(math.isfinite(v) for v in (xc, yc, w, h)):
                    add("error", "MALFORMED_BBOX", where, "non-finite coordinate")
                    continue
                if w <= 0 or h <= 0:
                    add("error", "ZERO_AREA_BBOX", where, f"width/height must be > 0 (w={w}, h={h})")
                    continue
                x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                if not all(0 <= v <= 1 for v in (xc, yc, w, h)) or x1 < -_EPS or y1 < -_EPS or x2 > 1 + _EPS or y2 > 1 + _EPS:
                    add("error", "BBOX_OUT_OF_BOUNDS", where, f"box ({x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}) outside image")
                    continue
                if w * width < 2 or h * height < 2:
                    add("warning", "TINY_BBOX", where, f"box smaller than 2 px ({w * width:.1f}x{h * height:.1f})")
                if line in seen_lines:
                    add("warning", "DUPLICATE_ANNOTATION", where, "identical box repeated")
                    continue
                seen_lines.add(line)
                cls = int(cls_f)
                for other_cls, other in boxes:
                    if other_cls == cls and _iou((x1, y1, x2, y2), other) > 0.95:
                        add("warning", "SUSPICIOUS_OVERLAP", where, "near-duplicate box of the same class (IoU > 0.95)")
                        break
                boxes.append((cls, (x1, y1, x2, y2)))
                counts[taxonomy.by_id(cls).name] += 1
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            if lbl_path.stem not in image_stems:
                add("error", "MISSING_IMAGE_FILE", lbl_path, "label file without a matching image")
        report.images[split] = n_images
        report.negatives[split] = negatives
        report.instances[split] = {name: counts.get(name, 0) for name in taxonomy.names}

    for digest, splits in hashes.items():
        if len(splits) > 1:
            add("warning", "CROSS_SPLIT_DUPLICATE", digest[:12], f"identical image in splits {sorted(splits)}")
    train = report.instances.get("train", {})
    for name, count in train.items():
        if count == 0:
            add("warning", "CLASS_WITHOUT_TRAINING_DATA", name,
                "no training instances: a model trained on this dataset cannot detect this class")
    return report
