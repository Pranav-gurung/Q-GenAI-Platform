"""Train a YOLOv8 defect detector on a YOLO-format dataset (Section A output).

Examples:
    # Fine-tune from COCO-pretrained YOLOv8n (downloads yolov8n.pt on first use)
    python -m training.train --data dataset/data.yaml --model yolov8n.pt --epochs 100

    # Train from scratch (no download) - needs far more data/epochs
    python -m training.train --data dataset/data.yaml --model yolov8n.yaml --epochs 300

The best checkpoint is copied to models/<output>.pt together with a
<output>.json metadata file. Register and activate it in the admin UI (or set
QGENAI_MODEL_WEIGHTS=<output>.pt) - the inference API does not change.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from training.validate_dataset import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def metrics_dict(m) -> dict:
    """Extract headline metrics from an Ultralytics DetMetrics object."""
    box = m.box
    return {"precision": float(box.mp), "recall": float(box.mr),
            "mAP50": float(box.map50), "mAP50_95": float(box.map)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/data.yaml", type=Path)
    ap.add_argument("--model", default="yolov8n.pt",
                    help="yolov8{n,s,m,l,x}.pt (pretrained) or .yaml (from scratch)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto batch size (GPU only)")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | 0,1 | mps")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=50, help="early-stopping patience (epochs)")
    ap.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train"))
    ap.add_argument("--name", default="defects")
    ap.add_argument("--output", default="defect_model", help="file name (no extension) in models/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Validating dataset {args.data} ...")
    rep = validate_dataset(args.data)
    for w in rep["warnings"]:
        print("WARNING:", w)
    if not rep["valid"]:
        for e in rep["errors"]:
            print("ERROR:", e)
        print("Dataset is invalid - fix the errors above before training.")
        return 1
    names = rep["class_names"]
    print(f"Classes ({len(names)}): {names}")

    from ultralytics import YOLO  # heavy import only after validation succeeds

    device = pick_device(args.device)
    print(f"Training {args.model} on device={device} for {args.epochs} epochs ...")
    model = YOLO(args.model)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, workers=args.workers, patience=args.patience, project=args.project,
                name=args.name, exist_ok=False, seed=args.seed, deterministic=True, plots=True)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"ERROR: no best.pt found in {save_dir}")
        return 1

    best_model = YOLO(str(best))
    results = {"val": metrics_dict(best_model.val(data=str(args.data.resolve()), split="val",
                                                   imgsz=args.imgsz, device=device, plots=False))}
    if "test" in rep["splits"] and rep["splits"]["test"]["images"]:
        results["test"] = metrics_dict(best_model.val(data=str(args.data.resolve()), split="test",
                                                       imgsz=args.imgsz, device=device, plots=False))

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    out = models_dir / f"{args.output}.pt"
    shutil.copy2(best, out)
    meta = {
        "weights_file": out.name,
        "class_names": names,
        "base_model": args.model,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "device": device,
        "dataset": str(args.data.resolve()),
        "dataset_summary": {k: {kk: v[kk] for kk in ("images", "instances", "per_class")}
                            for k, v in rep["splits"].items()},
        "metrics": results,  # measured on this run - nothing is hard-coded
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(save_dir),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved {out}\nNext: register + activate it under 'Policy & models' (admin), "
          f"or set QGENAI_MODEL_WEIGHTS={out.name} and restart the API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
