"""Draw REAL CV detections (boxes, defect type, confidence, severity) on an image.

    python -m ml.cv.visualize --input image.jpg --output annotated.jpg              # runs inference
    python -m ml.cv.visualize --input image.jpg --result predict.json --output a.jpg  # from predict output

Only boxes produced by the model are drawn; this tool never invents detections.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from backend.domain.enums import Severity
from backend.schemas.results import CVResult, Detection
from ml.cv.preprocessing import decode_image

SEVERITY_COLORS = {Severity.LOW: (46, 160, 67), Severity.MEDIUM: (219, 171, 9),
                   Severity.HIGH: (230, 100, 20), Severity.CRITICAL: (215, 30, 40), None: (120, 120, 255)}


def draw_detections(image: Image.Image, detections: list[Detection], header: str | None = None) -> Image.Image:
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    width = max(2, round(max(out.size) / 300))
    font = ImageFont.load_default(size=max(12, round(max(out.size) / 45)))
    for det in detections:
        if det.bbox is None:
            continue
        color = SEVERITY_COLORS.get(det.severity, SEVERITY_COLORS[None])
        b = det.bbox
        d.rectangle([b.x_min, b.y_min, b.x_max, b.y_max], outline=color, width=width)
        text = f"{det.defect_type.value} {det.confidence:.2f} {det.severity.value if det.severity else ''}".strip()
        tb = d.textbbox((0, 0), text, font=font)
        ty = b.y_min - (tb[3] - tb[1]) - 6 if b.y_min > (tb[3] - tb[1]) + 6 else b.y_max + 2
        d.rectangle([b.x_min, ty, b.x_min + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 6], fill=color)
        d.text((b.x_min + 3, ty + 2), text, fill=(255, 255, 255), font=font)
    if header:
        tb = d.textbbox((0, 0), header, font=font)
        d.rectangle([0, 0, tb[2] - tb[0] + 10, tb[3] - tb[1] + 10], fill=(0, 0, 0))
        d.text((5, 4), header, fill=(255, 255, 255), font=font)
    return out


def render(image_bytes: bytes, cv: CVResult) -> Image.Image:
    header = None
    if cv.inference:
        header = (f"{cv.inference.model_name}:{cv.inference.model_version}  {len(cv.detections)} det  "
                  f"{cv.inference.inference_time_ms:.0f} ms {cv.inference.inference_device}")
    return draw_detections(Image.fromarray(decode_image(image_bytes)), cv.detections, header)


def main(argv: list[str] | None = None) -> int:
    from ml.cv.cli import add_model_args, image_input_from_file, settings_from_args

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--result", type=Path, help="JSON from `ml.cv.predict` (skips inference)")
    add_model_args(p)
    args = p.parse_args(argv)

    if args.result:
        payload = json.loads(args.result.read_text())
        cv = CVResult.model_validate(payload.get("cv", payload))
    else:
        from ml.cv.yolo_engine import YoloCVEngine
        settings = settings_from_args(args)
        cv = YoloCVEngine(settings).detect(image_input_from_file(args.input, settings))
    if cv.status.value != "COMPLETED":
        print(f"no visualization: CV status {cv.status.value}: {cv.message}", file=sys.stderr)
        return 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args.input.read_bytes(), cv).save(args.output)
    print(f"wrote {args.output} ({len(cv.detections)} detections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
