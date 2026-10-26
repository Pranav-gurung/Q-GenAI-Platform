"""Optional: export trained weights to ONNX for faster CPU inference.

    python -m training.export_onnx --weights models/defect_model.pt
    # -> models/defect_model.onnx ; then set QGENAI_MODEL_WEIGHTS=defect_model.onnx
    #    (or register it in the admin UI). Requires `onnx` + `onnxruntime`.
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    if not args.weights.is_file():
        print(f"ERROR: weights not found: {args.weights}")
        return 1
    from ultralytics import YOLO

    out = YOLO(str(args.weights)).export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=False)
    print(f"Exported: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
