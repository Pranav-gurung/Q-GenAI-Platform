"""Optional: export a trained model to ONNX (future deployment artifact; not used by the Part 2 engine).

    python -m ml.cv.export --model artifacts/cv/<version>/model/best.pt [--imgsz 416] [--check]

Requires `pip install -e '.[cv-export]'` (onnx). --check runs one dummy inference with onnxruntime
(if installed) to verify the exported graph loads and returns the expected output shape.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from backend.core.config import get_settings

    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=s.cv_model_path)
    p.add_argument("--imgsz", type=int, default=None, help="default: training size")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("onnx is not installed: pip install -e '.[cv-export]'", file=sys.stderr)
        return 2
    from ml.cv._ultralytics import import_ultralytics
    import_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(args.model), task="detect")
    imgsz = args.imgsz or int(model.ckpt.get("train_args", {}).get("imgsz", 640))
    path = Path(model.export(format="onnx", imgsz=imgsz, opset=args.opset, dynamic=False, simplify=False))
    print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB, imgsz={imgsz})")
    if args.check:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed; skipping check", file=sys.stderr)
            return 0
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.run(None, {inp.name: np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)})
        print(f"onnxruntime check OK: input {inp.shape} -> output {out[0].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
