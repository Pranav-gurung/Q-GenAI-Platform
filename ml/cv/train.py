"""Reproducible YOLO training -> versioned model artifact.

    python -m ml.cv.train --data data/cv/synthetic-v1/dataset.yaml --version qgenai-yolov8n-synthetic-v1 \
        [--base yolov8n.pt|yolov8n.yaml] [--epochs 40] [--imgsz 416] [--batch 16] [--lr0 0.01] \
        [--device auto] [--seed 42] [--out artifacts/cv]

Steps: validate dataset (abort on errors) -> train (Ultralytics, deterministic, seeded, moderate
augmentation from preprocessing.TRAINING_AUGMENTATION) -> evaluate on val (+ test if present)
-> write artifacts/cv/<version>/{model/best.pt, metadata.json, metrics.json}.

The artifact records whether it was trained on SYNTHETIC data (dataset MANIFEST.json). Metrics of a
synthetic-trained model are demo metrics, not real-world accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ml.cv.dataset import validate_dataset
from ml.cv.preprocessing import TRAINING_AUGMENTATION
from ml.cv.taxonomy import load_taxonomy


def _describe_base(base: str) -> dict:
    """Name + content hash of the starting checkpoint (no machine-specific absolute paths)."""
    path = Path(base)
    info = {"name": path.name}
    if path.is_file():
        info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return info


def main(argv: list[str] | None = None) -> int:
    from backend.core.config import get_settings

    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/cv/synthetic-v1/dataset.yaml"))
    p.add_argument("--version", required=True, help="model version id, e.g. qgenai-yolov8n-synthetic-v1")
    p.add_argument("--base", default="yolov8n.pt", help="starting checkpoint (.pt, pretrained) or architecture (.yaml, scratch)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default=s.cv_device)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("artifacts/cv"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    taxonomy = load_taxonomy(s.cv_taxonomy_path)
    out_dir = args.out / args.version
    if out_dir.exists() and not args.overwrite:
        print(f"{out_dir} exists; choose a new --version or pass --overwrite", file=sys.stderr)
        return 2

    print(f"[1/4] validating dataset {args.data}")
    report = validate_dataset(args.data, taxonomy)
    if not report.ok:
        for issue in report.errors[:20]:
            print(f"  ERROR {issue.code}: {issue.path}: {issue.message}", file=sys.stderr)
        print("dataset invalid; run `python -m ml.cv.validate_dataset` for the full report", file=sys.stderr)
        return 1
    manifest_path = args.data.parent / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    synthetic = bool(manifest.get("synthetic"))

    from ml.cv._ultralytics import import_ultralytics
    ultralytics = import_ultralytics()
    import torch
    from ultralytics import YOLO

    from ml.cv.model import resolve_device

    device = resolve_device(args.device)
    params = {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "lr0": args.lr0,
              "patience": args.patience, "seed": args.seed, "deterministic": True, "device": device,
              "workers": args.workers, "optimizer": "AdamW" if args.lr0 < 0.005 else "SGD"}
    print("[2/4] training", json.dumps({"base": args.base, **params, "augmentation": TRAINING_AUGMENTATION}))
    print(f"      classes: {taxonomy.names}")
    print(f"      data: {'SYNTHETIC / DEMO' if synthetic else 'non-synthetic'}; train instances: "
          f"{report.instances.get('train')}")
    run_dir = out_dir / "runs"
    started = time.perf_counter()
    model = YOLO(args.base, task="detect")
    model.train(data=str(args.data.resolve()), project=str(run_dir.resolve()), name="train", exist_ok=True,
                plots=False, verbose=False, val=True, **params, **TRAINING_AUGMENTATION)
    duration = time.perf_counter() - started
    best = Path(model.trainer.best)
    if not best.is_file():
        print("training produced no best.pt", file=sys.stderr)
        return 1
    (out_dir / "model").mkdir(parents=True, exist_ok=True)
    weights = out_dir / "model" / "best.pt"
    shutil.copy2(best, weights)

    print("[3/4] evaluating")
    from ml.cv.evaluate import artifact_metrics
    metrics = artifact_metrics(weights, args.data, [sp for sp in ("val", "test") if report.images.get(sp)],
                               imgsz=args.imgsz, device=device, deploy_conf=s.cv_confidence_threshold,
                               model_version=args.version, synthetic=synthetic)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    train_counts = report.instances.get("train", {})
    metadata = {
        "model_name": Path(args.base).stem, "model_version": args.version, "task": "detect",
        "model_framework": f"ultralytics {ultralytics.__version__} / torch {torch.__version__}",
        "base_checkpoint": _describe_base(args.base), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_provenance": "synthetic_demo" if synthetic else "unspecified",
        "dataset": {"yaml": str(args.data), "manifest": manifest or None, "images": report.images,
                    "train_instances": train_counts},
        "classes": [{"id": c.id, "name": c.name, "defect_type": c.defect_type.value, "data_status": c.data_status,
                     "train_instances": train_counts.get(c.name, 0), "trained": train_counts.get(c.name, 0) > 0}
                    for c in taxonomy.classes],
        "training": {**params, "base": Path(args.base).name, "augmentation": TRAINING_AUGMENTATION,
                     "duration_s": round(duration, 1), "python": platform.python_version(),
                     "host_cpu": platform.processor() or platform.machine()},
        "artifact": {"weights": "model/best.pt", "size_bytes": weights.stat().st_size},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("[4/4] done")
    for split, m in metrics["splits"].items():
        print(f"  {split}: {json.dumps(m['detection'])}  image-level@{s.cv_confidence_threshold}: "
              f"{json.dumps({k: m['image_level'][k] for k in ('precision', 'recall', 'flag_rate')})}")
    print(f"  training time: {duration / 60:.1f} min on {device}")
    print(f"  artifact: {weights}")
    print(f"  use it:   CV_ENGINE=yolov8 CV_MODEL_PATH={weights}")
    if synthetic:
        print("  NOTE: trained and evaluated on SYNTHETIC / DEMO data only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
