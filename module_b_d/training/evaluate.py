"""Evaluate trained weights on a dataset split and save the measured metrics.

    python -m training.evaluate --weights models/defect_model.pt --data dataset/data.yaml --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from training.train import PROJECT_ROOT, metrics_dict, pick_device
from training.validate_dataset import validate_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", default="dataset/data.yaml", type=Path)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf is standard for mAP")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not args.weights.is_file():
        print(f"ERROR: weights not found: {args.weights}")
        return 1
    rep = validate_dataset(args.data)
    if not rep["valid"]:
        print("ERROR: dataset invalid:", *rep["errors"], sep="\n  ")
        return 1
    if args.split not in rep["splits"] or not rep["splits"][args.split]["images"]:
        print(f"ERROR: split '{args.split}' has no images.")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    m = model.val(data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz, conf=args.conf,
                  device=pick_device(args.device), plots=True,
                  project=str(PROJECT_ROOT / "runs" / "eval"), name=args.split)
    names = model.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": float(p), "recall": float(r),
                                    "mAP50": float(ap50), "mAP50_95": float(ap)}
    result = {"weights": str(args.weights), "split": args.split, "overall": metrics_dict(m),
              "per_class": per_class, "evaluated_at": datetime.now(timezone.utc).isoformat()}
    out = Path(m.save_dir) / "metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
