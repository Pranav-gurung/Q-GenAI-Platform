"""Generate a TINY synthetic dataset ONLY to smoke-test the training pipeline.

    python -m training.make_smoke_dataset --out dataset_smoke --n 60

!!! This is NOT the Section A dataset and NOT a manufacturing defect dataset.
!!! It draws simple coloured lines ("scratch") and blobs ("rust_spot") on grey
!!! plates so you can verify validate -> train -> evaluate -> load -> inspect
!!! end-to-end on a laptop in a few minutes. A model trained on it must never
!!! be used to inspect real products.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASSES = ["scratch", "rust_spot"]


def make_image(rng: random.Random, size: int):
    base = rng.randint(110, 170)
    img = np.full((size, size, 3), base, np.uint8)
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 6, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    labels = []
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.5:  # scratch: thin dark line
            x1, y1 = rng.randint(20, size - 120), rng.randint(20, size - 120)
            x2, y2 = x1 + rng.randint(40, 100), y1 + rng.randint(-30, 100)
            y2 = min(max(y2, 10), size - 10)
            cv2.line(img, (x1, y1), (x2, y2), (40, 40, 40), rng.randint(2, 4))
            bx1, by1, bx2, by2 = min(x1, x2) - 4, min(y1, y2) - 4, max(x1, x2) + 4, max(y1, y2) + 4
            cls = 0
        else:  # rust spot: brown filled ellipse
            cx, cy = rng.randint(40, size - 40), rng.randint(40, size - 40)
            ax, ay = rng.randint(10, 30), rng.randint(10, 30)
            cv2.ellipse(img, (cx, cy), (ax, ay), rng.randint(0, 180), 0, 360, (30, 70, 140), -1)
            bx1, by1, bx2, by2 = cx - max(ax, ay), cy - max(ax, ay), cx + max(ax, ay), cy + max(ax, ay)
            cls = 1
        bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(size, bx2), min(size, by2)
        labels.append((cls, (bx1 + bx2) / 2 / size, (by1 + by2) / 2 / size,
                       (bx2 - bx1) / size, (by2 - by1) / size))
    return img, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dataset_smoke", type=Path)
    ap.add_argument("--n", type=int, default=60, help="total images")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    splits = {"train": int(args.n * 0.7), "val": int(args.n * 0.2)}
    splits["test"] = args.n - splits["train"] - splits["val"]
    for split, n in splits.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels = make_image(rng, args.size)
            cv2.imwrite(str(args.out / "images" / split / f"{split}_{i:04d}.jpg"), img)
            (args.out / "labels" / split / f"{split}_{i:04d}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))
    (args.out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(args.out.resolve()), "train": "images/train", "val": "images/val",
        "test": "images/test", "nc": len(CLASSES), "names": CLASSES}, sort_keys=False))
    print(f"Smoke-test dataset written to {args.out} ({splits}). NOT for real inspections.")


if __name__ == "__main__":
    main()
