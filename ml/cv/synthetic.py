"""SYNTHETIC / DEMO DATA generator for the CV pipeline.

    python -m ml.cv.synthetic --out data/cv/synthetic-v1 --images 600 --seed 42

Draws simple dark-store-like scenes (cartons, cans and pouches on a shelf/tote background) and
injects procedurally drawn defects with exact bounding boxes. It exists so the full CV stack
(dataset validation -> training -> evaluation -> inference -> pipeline) can be built and
demonstrated without a real dataset.

!! Images are cartoon-like. A model trained on them demonstrates the ENGINEERING pipeline; its
!! metrics say nothing about performance on real product photos.

Classes generated (see taxonomy.yaml data_status=synthetic_demo): torn_packaging, leakage,
dented_container, damaged_box, missing_label. All other taxonomy classes get no examples.
Images without defects get an empty label file (explicit negatives). Fully deterministic for a
given --seed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

from ml.cv.taxonomy import Taxonomy, load_taxonomy

GENERATOR_VERSION = "1"
DISCLAIMER = "SYNTHETIC / DEMO DATA - procedurally generated; not representative of real product photos."

Box = tuple[float, float, float, float]

PRODUCT_COLORS = [(196, 154, 108), (176, 128, 84), (220, 60, 50), (40, 110, 190), (240, 200, 60),
                  (60, 160, 90), (235, 235, 230), (150, 70, 160), (250, 140, 40)]
LIQUIDS = [(245, 245, 240), (120, 70, 30), (230, 120, 20), (90, 150, 220), (200, 30, 40)]
BACKGROUNDS = [(120, 120, 125), (205, 195, 175), (60, 90, 140), (170, 175, 180), (90, 90, 85)]


@dataclass
class Annotation:
    class_name: str
    box: Box  # pixel xyxy


def _clip(box: Box, size: int) -> Box:
    x1, y1, x2, y2 = box
    return max(0.0, x1), max(0.0, y1), min(float(size), x2), min(float(size), y2)


def _jitter(rgb, rng, amount=25):
    return tuple(int(min(255, max(0, c + rng.randint(-amount, amount)))) for c in rgb)


def _shade(rgb, factor):
    return tuple(int(min(255, max(0, c * factor))) for c in rgb)


def _background(size: int, rng: random.Random, nrng: np.random.Generator) -> Image.Image:
    base = np.array(_jitter(rng.choice(BACKGROUNDS), rng, 20), dtype=np.float32)
    grad = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(1.0, 1.15), size, dtype=np.float32)[:, None, None]
    img = np.clip(base[None, None, :] * grad + nrng.normal(0, 8, (size, size, 3)), 0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(1, 3)):  # shelf edges / tote ribs
        y = rng.randint(0, size)
        d.line([(0, y), (size, y + rng.randint(-10, 10))], fill=_shade(tuple(base.astype(int)), 0.7), width=rng.randint(2, 6))
    return im


def _label(d: ImageDraw.ImageDraw, rng: random.Random, x1, y1, x2, y2) -> None:
    w, h = x2 - x1, y2 - y1
    lx1, lx2 = x1 + w * 0.12, x2 - w * 0.12
    ly1 = y1 + h * rng.uniform(0.3, 0.45)
    ly2 = ly1 + h * rng.uniform(0.25, 0.35)
    d.rectangle([lx1, ly1, lx2, ly2], fill=(250, 250, 245), outline=(30, 30, 30), width=1)
    d.rectangle([lx1, ly1, lx2, ly1 + (ly2 - ly1) * 0.3], fill=_jitter(rng.choice(PRODUCT_COLORS), rng))
    for i in range(3):  # "text" lines
        ty = ly1 + (ly2 - ly1) * (0.45 + 0.17 * i)
        d.line([(lx1 + 4, ty), (lx1 + 4 + (lx2 - lx1 - 8) * rng.uniform(0.4, 0.9), ty)], fill=(40, 40, 40), width=2)


def _draw_product(im: Image.Image, rng: random.Random, kind: str, box: Box, with_label: bool) -> None:
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = box
    color = _jitter(rng.choice(PRODUCT_COLORS), rng)
    if kind == "box":
        d.rectangle(box, fill=color, outline=_shade(color, 0.6), width=2)
        d.polygon([(x1, y1), (x1 + 8, y1 - 8), (x2 + 8, y1 - 8), (x2, y1)], fill=_shade(color, 1.15))  # top face
    elif kind == "can":
        r = (x2 - x1) / 2
        d.rectangle([x1, y1 + r * 0.25, x2, y2 - r * 0.25], fill=color)
        for i in range(int(x2 - x1)):  # cylinder shading
            f = 0.7 + 0.5 * math.sin(math.pi * i / max(1, (x2 - x1)))
            d.line([(x1 + i, y1 + r * 0.25), (x1 + i, y2 - r * 0.25)], fill=_shade(color, f))
        d.ellipse([x1, y1, x2, y1 + r * 0.5], fill=(200, 200, 205), outline=(120, 120, 125))
        d.ellipse([x1, y2 - r * 0.5, x2, y2], fill=_shade(color, 0.6))
    else:  # pouch
        d.rounded_rectangle(box, radius=(x2 - x1) * 0.18, fill=color, outline=_shade(color, 0.6), width=2)
        d.rectangle([x1 + 3, y1, x2 - 3, y1 + (y2 - y1) * 0.08], fill=_shade(color, 0.75))  # seal strip
    if with_label:
        _label(d, rng, x1, y1, x2, y2)


def _damaged_box(im, rng, box: Box) -> Box:
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = box
    s = min(x2 - x1, y2 - y1) * rng.uniform(0.35, 0.5)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    cx = x1 if corner in ("tl", "bl") else x2
    cy = y1 if corner in ("tl", "tr") else y2
    sx = 1 if corner in ("tl", "bl") else -1
    sy = 1 if corner in ("tl", "tr") else -1
    pts = [(cx, cy), (cx + sx * s, cy), (cx + sx * s * 0.55, cy + sy * s * 0.45), (cx, cy + sy * s)]
    d.polygon(pts, fill=(70, 50, 35))
    for _ in range(4):  # creases
        a = (cx + sx * rng.uniform(0, s), cy + sy * rng.uniform(0, s * 0.3))
        b = (cx + sx * rng.uniform(0, s * 0.3), cy + sy * rng.uniform(0, s))
        d.line([a, b], fill=(30, 20, 10), width=2)
    return (min(cx, cx + sx * s), min(cy, cy + sy * s), max(cx, cx + sx * s), max(cy, cy + sy * s))


def _torn(im, rng, box: Box) -> Box:
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    tx1 = x1 + w * rng.uniform(0.05, 0.35)
    tx2 = tx1 + w * rng.uniform(0.4, 0.6)
    ty = y1 + h * rng.uniform(0.15, 0.75)
    n = 9
    top = [(tx1 + (tx2 - tx1) * i / n, ty - rng.uniform(2, h * 0.08)) for i in range(n + 1)]
    bot = [(tx1 + (tx2 - tx1) * i / n, ty + rng.uniform(2, h * 0.08)) for i in reversed(range(n + 1))]
    d.polygon(top + bot, fill=(250, 248, 240), outline=(20, 20, 20))
    ys = [p[1] for p in top + bot]
    return (tx1 - 2, min(ys) - 2, tx2 + 2, max(ys) + 2)


def _dent(im, rng, box: Box) -> Box:
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dw, dh = w * rng.uniform(0.45, 0.7), h * rng.uniform(0.18, 0.3)
    dx = x1 + rng.uniform(0.05, 0.95 - dw / w) * w
    dy = y1 + h * rng.uniform(0.2, 0.75 - dh / h)
    d.ellipse([dx, dy, dx + dw, dy + dh], fill=(35, 35, 40))
    d.arc([dx + 2, dy + 2, dx + dw - 2, dy + dh - 2], 200, 340, fill=(235, 235, 235), width=3)
    d.line([(dx + dw * 0.2, dy + dh * 0.5), (dx + dw * 0.8, dy + dh * 0.55)], fill=(10, 10, 10), width=2)
    return (dx, dy, dx + dw, dy + dh)


def _leak(im, rng, box: Box, size: int) -> Box:
    x1, y1, x2, y2 = box
    liquid = rng.choice(LIQUIDS)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    w = x2 - x1
    pcx, pcy = (x1 + x2) / 2 + rng.uniform(-w * 0.2, w * 0.2), y2 + rng.uniform(4, 14)
    pw, ph = w * rng.uniform(0.8, 1.3), rng.uniform(16, 34)
    pts = []
    for k in range(14):
        a = 2 * math.pi * k / 14
        rr = rng.uniform(0.75, 1.15)
        pts.append((pcx + math.cos(a) * pw / 2 * rr, pcy + math.sin(a) * ph / 2 * rr))
    d.polygon(pts, fill=liquid + (215,))
    drip_x = [x1 + w * rng.uniform(0.2, 0.8) for _ in range(rng.randint(1, 3))]
    top = y2 - (y2 - y1) * rng.uniform(0.3, 0.6)
    for dx in drip_x:
        d.line([(dx, top), (dx + rng.uniform(-3, 3), pcy)], fill=liquid + (230,), width=rng.randint(4, 8))
        d.ellipse([dx - 5, top - 5, dx + 5, top + 5], fill=liquid + (230,))
    im.alpha_composite(overlay) if im.mode == "RGBA" else im.paste(Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB"))
    xs, ys = [p[0] for p in pts] + drip_x, [p[1] for p in pts] + [top - 5]
    return _clip((min(xs) - 5, min(ys), max(xs) + 5, max(ys)), size)


COMPATIBLE = {"box": ["damaged_box", "torn_packaging", "missing_label", "leakage"],
              "can": ["dented_container", "missing_label", "leakage"],
              "pouch": ["torn_packaging", "missing_label", "leakage"]}


def make_scene(rng: random.Random, nrng: np.random.Generator, size: int,
               defect_prob: float = 0.65) -> tuple[Image.Image, list[Annotation]]:
    im = _background(size, rng, nrng)
    n = rng.choice([1, 1, 2, 2, 3])
    cell = size / n
    anns: list[Annotation] = []
    for i in range(n):
        kind = rng.choice(["box", "can", "pouch"])
        if kind == "box":
            w, h = cell * rng.uniform(0.55, 0.8), size * rng.uniform(0.3, 0.5)
        elif kind == "can":
            w, h = cell * rng.uniform(0.35, 0.5), size * rng.uniform(0.3, 0.45)
        else:
            w, h = cell * rng.uniform(0.45, 0.7), size * rng.uniform(0.35, 0.55)
        w = min(w, size * 0.45)
        x1 = i * cell + rng.uniform(0.05, 0.95) * (cell - w)
        y1 = rng.uniform(size * 0.08, size * 0.85 - h)
        box = (x1, y1, x1 + w, y1 + h)
        defect = rng.choice(COMPATIBLE[kind]) if rng.random() < defect_prob else None
        _draw_product(im, rng, kind, box, with_label=defect != "missing_label")
        if defect == "missing_label":
            anns.append(Annotation(defect, _clip((box[0] - 2, box[1] - (10 if kind == "box" else 2), box[2] + (10 if kind == "box" else 2), box[3] + 2), size)))
        elif defect == "damaged_box":
            anns.append(Annotation(defect, _clip(_damaged_box(im, rng, box), size)))
        elif defect == "torn_packaging":
            anns.append(Annotation(defect, _clip(_torn(im, rng, box), size)))
        elif defect == "dented_container":
            anns.append(Annotation(defect, _clip(_dent(im, rng, box), size)))
        elif defect == "leakage":
            anns.append(Annotation(defect, _leak(im, rng, box, size)))
    if rng.random() < 0.15:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))  # mild camera softness
    return im.convert("RGB"), anns


def to_yolo_lines(anns: list[Annotation], taxonomy: Taxonomy, size: int) -> list[str]:
    lines = []
    for a in anns:
        x1, y1, x2, y2 = a.box
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cls = taxonomy.by_name(a.class_name).id
        lines.append(f"{cls} {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} "
                     f"{(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}")
    return lines


def generate(out: Path, n_images: int, seed: int, size: int = 416, taxonomy: Taxonomy | None = None,
             split: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> dict:
    taxonomy = taxonomy or load_taxonomy()
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = Path(out)
    n_train = int(n_images * split[0])
    n_val = int(n_images * split[1])
    counts: dict[str, dict[str, int]] = {}
    for idx in range(n_images):
        part = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        im, anns = make_scene(rng, nrng, size)
        stem = f"synth_{seed}_{idx:05d}"
        im.save(out / "images" / part / f"{stem}.jpg", quality=92)
        lines = to_yolo_lines(anns, taxonomy, size)
        (out / "labels" / part / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        c = counts.setdefault(part, {})
        for a in anns:
            c[a.class_name] = c.get(a.class_name, 0) + 1
    data_yaml = {
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {c.id: c.name for c in taxonomy.classes},
    }
    header = f"# {DISCLAIMER}\n# generator=ml.cv.synthetic v{GENERATOR_VERSION} seed={seed} images={n_images} size={size}\n"
    (out / "dataset.yaml").write_text(header + yaml.safe_dump(data_yaml, sort_keys=False))
    manifest = {"disclaimer": DISCLAIMER, "synthetic": True, "generator": f"ml.cv.synthetic v{GENERATOR_VERSION}",
                "seed": seed, "images": n_images, "image_size": size, "taxonomy_version": taxonomy.version,
                "instances": counts}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the SYNTHETIC / DEMO CV dataset")
    p.add_argument("--out", type=Path, default=Path("data/cv/synthetic-v1"))
    p.add_argument("--images", type=int, default=600)
    p.add_argument("--size", type=int, default=416)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"refusing to overwrite non-empty {args.out}", file=sys.stderr)
        return 2
    manifest = generate(args.out, args.images, args.seed, args.size)
    print(DISCLAIMER)
    print(json.dumps(manifest["instances"], indent=2))
    print(f"dataset written to {args.out}/dataset.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
