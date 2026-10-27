"""Measure CV inference latency on a set of images.

    python -m ml.cv.benchmark --images data/cv/synthetic-v1/images/test [--limit 50] [--runs 1] [--json out.json]

Reports model forward-pass latency (inference_time_ms) and end-to-end engine latency
(decode + preprocess + inference + postprocess). Numbers are for THIS machine and device only.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

from ml.cv.dataset import IMAGE_EXTS


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summary(values: list[float]) -> dict:
    return {"mean_ms": round(statistics.fmean(values), 2), "median_ms": round(statistics.median(values), 2),
            "p95_ms": round(_pct(values, 0.95), 2), "min_ms": round(min(values), 2), "max_ms": round(max(values), 2)}


def main(argv: list[str] | None = None) -> int:
    from ml.cv.cli import add_model_args, image_input_from_file, settings_from_args
    from ml.cv.yolo_engine import YoloCVEngine

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True, help="directory of images")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--runs", type=int, default=1, help="passes over the image set")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--json", type=Path)
    add_model_args(p)
    args = p.parse_args(argv)
    settings = settings_from_args(args)

    paths = sorted(x for x in args.images.iterdir() if x.suffix.lower() in IMAGE_EXTS)[: args.limit]
    if not paths:
        print(f"no images in {args.images}", file=sys.stderr)
        return 2
    engine = YoloCVEngine(settings)
    if not engine.available:
        print(f"CV engine unavailable: {engine.health().get('error')}", file=sys.stderr)
        return 3
    inputs = [image_input_from_file(x, settings) for x in paths]
    for img in inputs[: args.warmup]:
        engine.detect(img)
    model_ms, total_ms, n_det = [], [], 0
    for _ in range(args.runs):
        for img in inputs:
            r = engine.detect(img)
            model_ms.append(r.inference.inference_time_ms)
            total_ms.append(r.processing_time_ms)
            n_det += len(r.detections)
    h = engine.health()
    report = {"images": len(paths), "runs": args.runs, "measurements": len(model_ms),
              "device": h["device"], "model_version": h["model_version"], "image_size": h["image_size"],
              "cpu": platform.processor() or platform.machine(), "detections_total": n_det,
              "model_inference": _summary(model_ms), "end_to_end": _summary(total_ms)}
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
