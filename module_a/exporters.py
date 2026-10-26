"""Export adapters. The canonical format (module_a.schema) is the source of truth; adapters are one-way.

Supported: ``coco`` (bbox detection), ``yolo`` (bbox detection), ``anomaly`` (folder layout for
anomaly-detection tooling). Segmentation masks are NOT implemented: asking for it raises a clear error.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .schema import ImageRecord

EXPORT_FORMATS = ("canonical", "coco", "yolo", "anomaly")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_coco(records_by_split: dict, images_dir_by_split: dict, class_map: dict, out_dir: Path) -> dict:
    """COCO bbox format: bbox=[x,y,width,height] (converted from canonical xyxy); category ids start at 1."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    cats = [{"id": i + 1, "name": n} for n, i in sorted(class_map.items(), key=lambda kv: kv[1])]
    for split, recs in records_by_split.items():
        images, anns, aid = [], [], 1
        for n, r in enumerate(recs, start=1):
            images.append({"id": n, "file_name": r.file_name, "width": r.width, "height": r.height})
            for a in r.annotations:
                if a.bbox is None:
                    raise NotImplementedError("COCO export of mask-only annotations is not implemented")
                x0, y0, x1, y1 = a.bbox
                anns.append({"id": aid, "image_id": n, "category_id": class_map[a.class_name] + 1,
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
                aid += 1
        p = out_dir / f"instances_{split}.json"
        p.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats,
                                 "info": {"note": "images are relative to images_dir",
                                          "images_dir": str(images_dir_by_split[split])}}, indent=1))
        written[split] = str(p)
    return written


def export_yolo(records_by_split: dict, images_dir_by_split: dict, class_map: dict, out_dir: Path) -> dict:
    """YOLO txt: ``class cx cy w h`` normalised to [0,1]; class ids are 0-based canonical ids."""
    out_dir = Path(out_dir)
    for split, recs in records_by_split.items():
        for r in recs:
            _link_or_copy(Path(images_dir_by_split[split]) / r.file_name, out_dir / "images" / split / r.file_name)
            lines = []
            for a in r.annotations:
                if a.bbox is None:
                    raise NotImplementedError("YOLO-detect export of mask-only annotations is not implemented")
                x0, y0, x1, y1 = a.bbox
                lines.append(f"{class_map[a.class_name]} {(x0 + x1) / 2 / r.width:.6f} {(y0 + y1) / 2 / r.height:.6f} "
                             f"{(x1 - x0) / r.width:.6f} {(y1 - y0) / r.height:.6f}")
            (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / split / (Path(r.file_name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    names = "\n".join(f"  {i}: {n}" for n, i in sorted(class_map.items(), key=lambda kv: kv[1]))
    (out_dir / "data.yaml").write_text(f"path: {Path(out_dir).resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    return {"data_yaml": str(out_dir / "data.yaml")}


def export_anomaly(records_by_split: dict, images_dir_by_split: dict, class_map: dict, out_dir: Path) -> dict:
    """Folder layout: <split>/good/*.png and <split>/<primary_class>/*.png (primary = first annotation's class).

    NOTE: anomaly detectors normally train on defect-free images only; the layout does not enforce that.
    """
    out_dir = Path(out_dir)
    counts: dict = {}
    for split, recs in records_by_split.items():
        for r in recs:
            sub = "good" if not r.annotations else r.annotations[0].class_name
            _link_or_copy(Path(images_dir_by_split[split]) / r.file_name, out_dir / split / sub / r.file_name)
            counts[f"{split}/{sub}"] = counts.get(f"{split}/{sub}", 0) + 1
    return {"layout": "split/{good|<class>}/image", "counts": counts}


def export(fmt: str, records_by_split, images_dir_by_split, class_map, out_dir) -> dict:
    if fmt == "canonical":
        return {}
    if fmt == "segmentation":
        raise NotImplementedError("segmentation-mask export is not implemented in this version")
    fn = {"coco": export_coco, "yolo": export_yolo, "anomaly": export_anomaly}.get(fmt)
    if fn is None:
        raise ValueError(f"unknown export format {fmt!r}; choose from {EXPORT_FORMATS}")
    return fn(records_by_split, images_dir_by_split, class_map, out_dir)
