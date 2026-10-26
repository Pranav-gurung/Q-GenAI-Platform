"""Reproducible, group-aware, class-aware train/val/test splitting (performed BEFORE augmentation).

Images are first clustered so that anything that must not be separated stays together:
  * images sharing a ``group_id`` (physical part / capture session / source image)
  * byte-identical files, and perceptual near-duplicates (dHash distance <= threshold)
Whole clusters are assigned to one split. Assignment is a seeded greedy heuristic that matches
per-class instance targets; it is NOT optimal and small datasets will deviate from the requested
ratios (this is reported).

Limitation: perceptual hashes catch only mild re-encodings/noise, not rotated/cropped/relit views of
the same part. Accurate ``group_id`` metadata is the real leakage protection.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .hashing import hamming_hex
from .schema import ImageRecord, Issue, load_canonical

SPLITS = ("train", "val", "test")


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_clusters(records: list, near_dup_threshold: int = 4) -> dict:
    """Return {cluster_id: [image_id,...]} (deterministic)."""
    uf = _UF([r.image_id for r in records])
    by_group, by_hash = defaultdict(list), defaultdict(list)
    for r in records:
        by_group[r.effective_group].append(r.image_id)
        if r.sha256:
            by_hash[r.sha256].append(r.image_id)
    for ids in list(by_group.values()) + list(by_hash.values()):
        for other in ids[1:]:
            uf.union(ids[0], other)
    hashed = [r for r in records if r.phash]
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            if hamming_hex(hashed[i].phash, hashed[j].phash) <= near_dup_threshold:
                uf.union(hashed[i].image_id, hashed[j].image_id)
    clusters: dict = defaultdict(list)
    for r in records:
        clusters[uf.find(r.image_id)].append(r.image_id)
    return {k: sorted(v) for k, v in sorted(clusters.items())}


def split_records(records: list, ratios=(0.70, 0.15, 0.15), seed: int = 42, near_dup_threshold: int = 4,
                  min_eval_images: int = 30, min_class_instances_per_split: int = 5) -> tuple:
    """Assign ``record.split`` in place. Returns (stats_dict, issues)."""
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6 or any(r <= 0 for r in ratios):
        raise ValueError(f"ratios must be 3 positive numbers summing to 1, got {ratios}")
    by_id = {r.image_id: r for r in records}
    clusters = build_clusters(records, near_dup_threshold)
    rng = random.Random(seed)
    cl = list(clusters.items())
    rng.shuffle(cl)
    cl.sort(key=lambda kv: -len(kv[1]))  # stable: big clusters first, random order among equals

    classes = sorted({a.class_name for r in records for a in r.annotations})
    total_cls = Counter(a.class_name for r in records for a in r.annotations)
    n = len(records)
    cur_img = {s: 0 for s in SPLITS}
    cur_cls = {s: Counter() for s in SPLITS}

    def _marginal(cur: float, add: float, target_ratio: float, total: float) -> float:
        """Change in squared fractional deviation from the target share when ``add`` is assigned."""
        if total <= 0:
            return 0.0
        return ((cur + add) / total - target_ratio) ** 2 - (cur / total - target_ratio) ** 2

    for _, ids in cl:
        add_cls = Counter(a.class_name for i in ids for a in by_id[i].annotations)
        best, best_cost = None, None
        for k, s in enumerate(SPLITS):
            cost = _marginal(cur_img[s], len(ids), ratios[k], n)
            for c in classes:
                cost += _marginal(cur_cls[s][c], add_cls[c], ratios[k], total_cls[c])
            if best_cost is None or cost < best_cost - 1e-12:
                best, best_cost = s, cost
        for i in ids:
            by_id[i].split = best
        cur_img[best] += len(ids)
        cur_cls[best].update(add_cls)

    issues: list = []
    stats = {"seed": seed, "ratios_requested": list(ratios), "n_clusters": len(clusters),
             "largest_cluster": max((len(v) for v in clusters.values()), default=0),
             "splits": {}}
    for s in SPLITS:
        imgs = [r for r in records if r.split == s]
        inst = Counter(a.class_name for r in imgs for a in r.annotations)
        img_cls = Counter(c for r in imgs for c in {a.class_name for a in r.annotations})
        stats["splits"][s] = {"n_images": len(imgs), "ratio_actual": round(len(imgs) / n, 4) if n else 0,
                              "instances_per_class": {c: inst[c] for c in classes},
                              "images_per_class": {c: img_cls[c] for c in classes},
                              "n_defect_free": sum(1 for r in imgs if not r.annotations)}
        if s != "train" and len(imgs) < min_eval_images:
            issues.append(Issue("small_eval_split", "warning",
                                f"{s} split has only {len(imgs)} images (< {min_eval_images}); metrics will be unreliable"))
        for c in classes:
            if inst[c] == 0:
                issues.append(Issue("class_missing_in_split", "warning", f"class {c!r} has no instances in {s}"))
            elif inst[c] < min_class_instances_per_split and s != "train":
                issues.append(Issue("class_rare_in_split", "warning",
                                    f"class {c!r} has only {inst[c]} instances in {s}"))
    if n and stats["largest_cluster"] / n > 0.3:
        issues.append(Issue("giant_cluster", "warning",
                            f"one group/duplicate cluster holds {stats['largest_cluster']} of {n} images; "
                            "splits cannot follow the requested ratios (check group_id / near-dup threshold)"))
    return stats, issues


def check_leakage(records: list, near_dup_threshold: int = 4) -> list:
    """Independent post-hoc audit of split assignments (originals and augmented variants)."""
    issues: list = []
    by_id = {r.image_id: r for r in records}
    grp_splits: dict = defaultdict(set)
    hash_splits: dict = defaultdict(set)
    for r in records:
        grp_splits[r.effective_group].add(r.split)
        if r.sha256:
            hash_splits[r.sha256].add(r.split)
        if r.source_image_id:
            src = by_id.get(r.source_image_id)
            if src is None or src.split != r.split:
                issues.append(Issue("augment_leak", "error",
                                    f"augmented image {r.image_id} is in split {r.split!r} but its source is not",
                                    r.image_id))
            if r.split != "train":
                issues.append(Issue("augment_outside_train", "error",
                                    f"augmented image {r.image_id} found in non-train split {r.split!r}", r.image_id))
    for g, s in grp_splits.items():
        if len(s) > 1:
            issues.append(Issue("group_leak", "error", f"group {g!r} spans splits {sorted(map(str, s))}"))
    for h, s in hash_splits.items():
        if len(s) > 1:
            issues.append(Issue("duplicate_leak", "error", f"identical file content spans splits {sorted(map(str, s))}",
                                details={"sha256": h}))
    originals = [r for r in records if r.phash and not r.source_image_id]
    for i in range(len(originals)):
        for j in range(i + 1, len(originals)):
            a, b = originals[i], originals[j]
            if a.split != b.split and hamming_hex(a.phash, b.phash) <= near_dup_threshold:
                issues.append(Issue("near_duplicate_leak", "error",
                                    f"near-duplicate images {a.image_id} ({a.split}) and {b.image_id} ({b.split})"))
    return issues


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Split a canonical annotation file (needs sha256/phash: run prepare_dataset first)")
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    ap.add_argument("--out", default="data/reports/split_stats.json")
    a = ap.parse_args(argv)
    records, _, _ = load_canonical(Path(a.canonical))
    stats, issues = split_records(records, tuple(float(x) for x in a.ratios.split(",")), a.seed)
    issues += check_leakage(records)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"stats": stats, "issues": [i.to_dict() for i in issues],
                                       "assignments": {r.image_id: r.split for r in records}}, indent=2))
    print(json.dumps(stats["splits"], indent=1))
    return 1 if any(i.severity == "error" for i in issues) else 0


if __name__ == "__main__":
    sys.exit(main())
