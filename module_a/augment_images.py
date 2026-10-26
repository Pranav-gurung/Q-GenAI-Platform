"""Seeded augmentation that transforms annotations together with pixels. TRAIN SPLIT ONLY.

Geometry (applied in this order): crop -> flips -> small rotation; then photometric/noise/blur.
Every geometric op updates the bounding boxes. Boxes under rotation become the axis-aligned
envelope of the rotated corners, which over-estimates tight extent for elongated defects
(e.g. diagonal scratches); keep ``rotation_max_deg`` small.

Flips are disabled by default: whether a flip is physically valid (text, handedness, directional
lighting, orientation-dependent defects) is a domain decision the dataset owner must make.
Mask annotations are not supported by the geometric ops (raises) -- see README limitations.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .hashing import dhash, sha256_file
from .schema import Annotation, ImageRecord


@dataclass
class AugmentConfig:
    seed: int = 42
    variants_per_image: int = 2
    rotation_max_deg: float = 8.0
    p_rotate: float = 0.5
    brightness_range: tuple = (0.85, 1.15)
    contrast_range: tuple = (0.85, 1.15)
    p_photometric: float = 0.8
    allow_hflip: bool = False
    allow_vflip: bool = False
    p_flip: float = 0.5
    p_crop: float = 0.3
    crop_min_scale: float = 0.8
    p_noise: float = 0.25
    noise_sigma_max: float = 6.0
    p_blur: float = 0.2
    blur_radius_max: float = 1.0
    min_visible_ratio: float = 0.6  # drop a box if less than this fraction survives clipping

    def to_dict(self) -> dict:
        return asdict(self)


def _rng_for(seed: int, image_id: str, k: int) -> tuple:
    n = int.from_bytes(hashlib.sha256(f"{seed}|{image_id}|{k}".encode()).digest()[:8], "big")
    return random.Random(n), np.random.default_rng(n)


# --- geometry helpers (all coordinates: continuous pixel-edge coordinates, y down) ---------------
def rotate_bbox(bbox: list, angle_deg: float, w: int, h: int) -> list:
    """Envelope of ``bbox`` after PIL ``Image.rotate(angle_deg)`` (counter-clockwise on screen, about the centre)."""
    cx, cy = w / 2.0, h / 2.0
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    xs, ys = [], []
    for x, y in ((bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])):
        dx, dy = x - cx, y - cy
        xs.append(cx + dx * c + dy * s)
        ys.append(cy - dx * s + dy * c)
    return [min(xs), min(ys), max(xs), max(ys)]


def clip_bbox(bbox: list, w: int, h: int) -> tuple:
    """Return (clipped_bbox_or_None, visible_ratio)."""
    area = max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)
    x0, y0, x1, y1 = max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], w), min(bbox[3], h)
    if x1 <= x0 or y1 <= y0 or area <= 0:
        return None, 0.0
    return [x0, y0, x1, y1], ((x1 - x0) * (y1 - y0)) / area


def _round_box(b: list) -> list:
    return [round(float(v), 2) for v in b]


def augment_image(img: Image.Image, annotations: list, cfg: AugmentConfig, rng: random.Random,
                  nprng: np.random.Generator) -> tuple:
    """Return (new_image, new_annotations, ops_log, n_dropped_annotations)."""
    if any(a.bbox is None for a in annotations):
        raise NotImplementedError("mask-only annotations are not supported by geometric augmentation")
    img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
    boxes = [list(a.bbox) for a in annotations]
    keep = [True] * len(boxes)
    ops: list = []
    dropped = 0

    def apply_clip(w, h):
        nonlocal dropped
        for i, b in enumerate(boxes):
            if not keep[i]:
                continue
            clipped, ratio = clip_bbox(b, w, h)
            if clipped is None or ratio < cfg.min_visible_ratio:
                keep[i] = False
                dropped += 1
            else:
                boxes[i] = clipped

    w, h = img.size
    if rng.random() < cfg.p_crop and cfg.crop_min_scale < 1.0:
        scale = rng.uniform(cfg.crop_min_scale, 1.0)
        cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        boxes = [[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0] for b in boxes]
        w, h = img.size
        apply_clip(w, h)
        ops.append({"op": "crop", "x0": x0, "y0": y0, "w": cw, "h": ch})
    if cfg.allow_hflip and rng.random() < cfg.p_flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
        ops.append({"op": "hflip"})
    if cfg.allow_vflip and rng.random() < cfg.p_flip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        boxes = [[b[0], h - b[3], b[2], h - b[1]] for b in boxes]
        ops.append({"op": "vflip"})
    if cfg.rotation_max_deg > 0 and rng.random() < cfg.p_rotate:
        ang = rng.uniform(-cfg.rotation_max_deg, cfg.rotation_max_deg)
        img = img.rotate(ang, resample=Image.BILINEAR, expand=False, fillcolor=0 if img.mode == "L" else (0, 0, 0))
        boxes = [rotate_bbox(b, ang, w, h) for b in boxes]
        apply_clip(w, h)
        ops.append({"op": "rotate", "angle_deg": round(ang, 3)})
    if rng.random() < cfg.p_photometric:
        bf, cf = rng.uniform(*cfg.brightness_range), rng.uniform(*cfg.contrast_range)
        img = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(bf)).enhance(cf)
        ops.append({"op": "brightness_contrast", "brightness": round(bf, 3), "contrast": round(cf, 3)})
    if rng.random() < cfg.p_blur and cfg.blur_radius_max > 0:
        r = rng.uniform(0.3, cfg.blur_radius_max)
        img = img.filter(ImageFilter.GaussianBlur(r))
        ops.append({"op": "blur", "radius": round(r, 3)})
    if rng.random() < cfg.p_noise and cfg.noise_sigma_max > 0:
        sigma = rng.uniform(1.0, cfg.noise_sigma_max)
        arr = np.asarray(img).astype(np.float32)
        arr += nprng.normal(0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        ops.append({"op": "noise", "sigma": round(sigma, 3)})

    new_anns = [Annotation(a.class_name, _round_box(b)) for a, b, k in zip(annotations, boxes, keep) if k]
    return img, new_anns, ops, dropped


def augment_records(records: list, image_root: Path, out_dir: Path, cfg: AugmentConfig,
                    allowed_split: str = "train") -> tuple:
    """Create augmented variants for ``records`` (all must be in ``allowed_split``).

    Returns (new_records, log). Variants inherit the source group and split; originals are untouched.
    """
    bad = [r.image_id for r in records if r.split != allowed_split]
    if bad:
        raise ValueError(f"augmentation is restricted to the {allowed_split!r} split; got records from other splits: {bad[:5]}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_records, log = [], []
    for rec in records:
        if rec.source_image_id:  # never augment an augmented image
            continue
        with Image.open(Path(image_root) / rec.file_name) as im:
            im.load()
            src = im.copy()
        for k in range(cfg.variants_per_image):
            rng, nprng = _rng_for(cfg.seed, rec.image_id, k)
            new_img, anns, ops, dropped = augment_image(src, rec.annotations, cfg, rng, nprng)
            if rec.annotations and not anns:
                log.append({"source": rec.image_id, "variant": k, "status": "rejected",
                            "reason": "all annotations fell outside the augmented view"})
                continue
            vid = f"{rec.image_id}__aug{k:02d}"
            fname = f"{vid}.png"
            new_img.save(out_dir / fname, format="PNG")
            new_records.append(ImageRecord(
                image_id=vid, file_name=fname, width=new_img.size[0], height=new_img.size[1],
                group_id=rec.effective_group, annotations=anns, split=allowed_split,
                source_image_id=rec.image_id, aug_ops=ops,
                sha256=sha256_file(out_dir / fname), phash=dhash(new_img)))
            log.append({"source": rec.image_id, "variant": k, "status": "ok", "ops": ops,
                        "annotations_dropped": dropped})
    return new_records, log
