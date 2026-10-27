"""Run the real CV engine (and the image quality gate) on one image and print JSON.

    python -m ml.cv.predict --image path/to/image.jpg [--save-visualization out.jpg] [--model ... --device cpu]

Exit codes: 0 inference ran, 1 invalid image, 3 CV engine unavailable (e.g. no model artifact).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.core.exceptions import InvalidImageError


def main(argv: list[str] | None = None) -> int:
    from ml.cv.cli import add_model_args, image_input_from_file, settings_from_args
    from ml.cv.quality import BasicImageQualityGate
    from ml.cv.visualize import render
    from ml.cv.yolo_engine import YoloCVEngine

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--save-visualization", type=Path)
    p.add_argument("--no-quality", action="store_true", help="skip the image quality gate")
    add_model_args(p)
    args = p.parse_args(argv)
    settings = settings_from_args(args)

    try:
        image = image_input_from_file(args.image, settings)
    except InvalidImageError as exc:
        print(json.dumps({"image": str(args.image), "error": {"code": exc.code, "message": exc.message}}))
        return 1
    out: dict = {"image": str(args.image)}
    if not args.no_quality:
        out["quality"] = BasicImageQualityGate(settings).assess(image).model_dump(mode="json")
    cv = YoloCVEngine(settings).detect(image)
    out["cv"] = cv.model_dump(mode="json")
    print(json.dumps(out, indent=2))
    if cv.status.value != "COMPLETED":
        return 3
    if args.save_visualization:
        args.save_visualization.parent.mkdir(parents=True, exist_ok=True)
        render(image.load_bytes(), cv).save(args.save_visualization)
        print(f"visualization: {args.save_visualization}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
