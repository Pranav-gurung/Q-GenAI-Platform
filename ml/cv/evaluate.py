"""Evaluate a trained detector on a dataset split and write machine-readable metrics.

    python -m ml.cv.evaluate --model artifacts/cv/<version>/model/best.pt \
        --data data/cv/synthetic-v1/dataset.yaml --split test [--out metrics.json]

Reports
  detection   precision / recall (at the F1-optimal confidence, Ultralytics convention), mAP@0.5, mAP@0.5:0.95
  per_class   instances, P, R, AP50, AP50-95. Classes with 0 instances are marked evaluated=false:
              there is NO evidence about them, regardless of what the model outputs.
  confusion   confusion matrix at the DEPLOYMENT threshold (rows = predicted, cols = true; last row =
              missed objects, last column = false positives)
  image_level at the DEPLOYMENT threshold (CV_CONFIDENCE_THRESHOLD): does the image contain any
              defect? -> TP/FP/FN/TN, precision, recall, flag rate. This is what drives review/fail rates.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

from ml.cv.dataset import IMAGE_EXTS, dataset_root


def _dataset_root(data_yaml: Path) -> tuple[Path, dict]:
    cfg = yaml.safe_load(data_yaml.read_text())
    return dataset_root(data_yaml, cfg), cfg


def image_level(model, data_yaml: Path, split: str, conf: float, imgsz: int, device: str) -> dict:
    root, cfg = _dataset_root(data_yaml)
    img_dir = root / cfg[split]
    lbl_dir = root / str(cfg[split]).replace("images", "labels", 1)
    tp = fp = fn = tn = 0
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label = lbl_dir / f"{img.stem}.txt"
        has_gt = label.is_file() and bool(label.read_text().strip())
        r = model.predict(source=str(img), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        has_pred = r.boxes is not None and len(r.boxes) > 0
        tp += has_gt and has_pred
        fn += has_gt and not has_pred
        fp += (not has_gt) and has_pred
        tn += (not has_gt) and not has_pred
    n = tp + fp + fn + tn
    return {"threshold": conf, "images": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "flag_rate": round((tp + fp) / n, 4) if n else None}


def evaluate(weights: Path, data_yaml: Path, split: str, *, imgsz: int | None, device: str,
             deploy_conf: float, batch: int = 16) -> dict:
    from ml.cv._ultralytics import import_ultralytics
    import_ultralytics()
    from ultralytics import YOLO

    from ml.cv.model import resolve_device

    device = resolve_device(device)
    model = YOLO(str(weights), task="detect")
    imgsz = imgsz or int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 640))
    with tempfile.TemporaryDirectory() as tmp:
        # Pass 1: standard detection metrics (P/R/mAP need the full low-confidence ranking).
        m = model.val(data=str(data_yaml.resolve()), split=split, imgsz=imgsz, batch=batch, device=device,
                      plots=False, verbose=False, project=tmp, name="val", conf=0.001, iou=0.6)
        # Pass 2: confusion matrix at the DEPLOYMENT threshold. Ultralytics fills it at the validation conf
        # and only with plots=True (plot files go to the temp dir).
        m_deploy = model.val(data=str(data_yaml.resolve()), split=split, imgsz=imgsz, batch=batch, device=device,
                             plots=True, verbose=False, project=tmp, name="cm", conf=deploy_conf, iou=0.6)
    names = model.names
    box = m.box
    per_class = {}
    nt = getattr(m, "nt_per_class", None)
    instances = [int(x) for x in nt] if nt is not None else []
    index = {int(c): i for i, c in enumerate(box.ap_class_index)}
    for cid, name in names.items():
        n_inst = int(instances[cid]) if cid < len(instances) else 0
        if cid in index and n_inst > 0:
            i = index[cid]
            per_class[name] = {"instances": n_inst, "evaluated": True, "precision": round(float(box.p[i]), 4),
                               "recall": round(float(box.r[i]), 4), "AP50": round(float(box.ap50[i]), 4),
                               "AP50_95": round(float(box.ap[i]), 4)}
        else:
            per_class[name] = {"instances": n_inst, "evaluated": False}
    cm = getattr(m_deploy, "confusion_matrix", None)
    matrix = cm.matrix.astype(int).tolist() if cm is not None and cm.matrix.sum() > 0 else None  # never a fake zero matrix
    return {
        "split": split, "dataset": str(data_yaml), "image_size": imgsz, "device": device,
        "detection": {"precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
                      "mAP50": round(float(box.map50), 4), "mAP50_95": round(float(box.map), 4)},
        "per_class": per_class,
        "confusion_matrix": {"labels": [names[i] for i in sorted(names)] + ["background"],
                             "rows": "predicted", "cols": "true", "confidence_threshold": deploy_conf, "iou": 0.45,
                             "note": "last row = missed objects (FN) per true class; "
                                     "last column = false positives per predicted class",
                             "matrix": matrix},
        "image_level": image_level(model, data_yaml, split, deploy_conf, imgsz, device),
    }


def artifact_metrics(weights: Path, data_yaml: Path, splits: list[str], *, imgsz: int | None, device: str,
                     deploy_conf: float, model_version: str, synthetic: bool) -> dict:
    """The metrics.json document stored next to a model artifact."""
    return {"model_version": model_version, "synthetic_data": synthetic,
            "disclaimer": ("Metrics on SYNTHETIC / DEMO data. They demonstrate the pipeline and say nothing about "
                           "accuracy on real product photos.") if synthetic else None,
            "splits": {split: evaluate(weights, data_yaml, split, imgsz=imgsz, device=device, deploy_conf=deploy_conf)
                       for split in splits}}


def main(argv: list[str] | None = None) -> int:
    from backend.core.config import get_settings

    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=s.cv_model_path)
    p.add_argument("--data", type=Path, default=Path("data/cv/synthetic-v1/dataset.yaml"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=s.cv_image_size)
    p.add_argument("--device", default=s.cv_device)
    p.add_argument("--conf", type=float, default=s.cv_confidence_threshold, help="deployment threshold")
    p.add_argument("--out", type=Path)
    p.add_argument("--artifact", type=Path, help="artifact dir: re-evaluate val+test and rewrite <dir>/metrics.json")
    args = p.parse_args(argv)
    if args.artifact:
        meta = json.loads((args.artifact / "metadata.json").read_text())
        manifest = (meta.get("dataset") or {}).get("manifest") or {}
        result = artifact_metrics(args.artifact / "model" / "best.pt", args.data, ["val", "test"], imgsz=args.imgsz,
                                  device=args.device, deploy_conf=args.conf, model_version=meta["model_version"],
                                  synthetic=bool(manifest.get("synthetic")))
        (args.artifact / "metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({s: r["detection"] for s, r in result["splits"].items()}, indent=2))
        return 0
    result = evaluate(args.model, args.data, args.split, imgsz=args.imgsz, device=args.device, deploy_conf=args.conf)
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
