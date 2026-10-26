"""Handover verification from Module A (Synthetic Data Engine) to Module B (YOLO Detection).

Verifies that a Module A YOLO export:
1. Fully satisfies Module B's dataset requirements (data.yaml, images, labels, bounds, class IDs).
2. Matches the dataset manifest class map and provenance (when manifest is present).
3. Is provably free of data leakage across splits (train, val, test) via content hashing (SHA-256)
   and group/part assignment checks.

Fails clearly and loudly when errors or leakages are detected.
Never silently modifies, re-labels, or discards data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from training.validate_dataset import IMG_EXT, load_names, validate_dataset


@dataclass
class HandoverReport:
    valid: bool
    data_yaml: str
    class_names: list[str]
    splits: dict
    manifest_verified: bool
    manifest_path: str | None
    leakage_free: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _extract_group_id(filename: str) -> str | None:
    """Extract common grouping tokens like part001, lotA, or base name before augmentation."""
    stem = Path(filename).stem
    # Strip augmentation suffixes like __aug01 or _aug2
    base = re.sub(r"(__aug\d+|_aug\d+)$", "", stem)
    # Check for partXXX or groupXXX prefix
    m = re.match(r"^(part\d+|lot[A-Za-z0-9]+|sample\d+)", base, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return base.lower() if base != stem else None


def verify_handover(
    data_yaml_or_dir: Path | str,
    manifest_path: Path | str | None = None,
) -> HandoverReport:
    """Verify that a Module A YOLO export is valid and leakage-free for Module B training."""
    errors: list[str] = []
    warnings: list[str] = []
    target = Path(data_yaml_or_dir).resolve()

    if target.is_dir():
        data_yaml = target / "data.yaml"
    else:
        data_yaml = target

    if not data_yaml.is_file():
        errors.append(f"data.yaml not found at {data_yaml}")
        return HandoverReport(
            valid=False,
            data_yaml=str(data_yaml),
            class_names=[],
            splits={},
            manifest_verified=False,
            manifest_path=None,
            leakage_free=False,
            errors=errors,
            warnings=warnings,
        )

    # 1. Run Module B dataset validation
    b_report = validate_dataset(data_yaml)
    errors.extend(b_report.get("errors", []))
    warnings.extend(b_report.get("warnings", []))
    class_names = b_report.get("class_names", [])
    splits_info = b_report.get("splits", {})

    # 2. Check data.yaml root path resolution
    try:
        cfg = yaml.safe_load(data_yaml.read_text()) or {}
        root = Path(cfg.get("path") or data_yaml.parent)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
    except Exception as exc:
        errors.append(f"Failed to read data.yaml: {exc}")
        root = data_yaml.parent

    # 3. Check for Manifest and Cross-Verify Class Map & Provenance
    manifest_file = None
    if manifest_path:
        manifest_file = Path(manifest_path).resolve()
    else:
        # Check standard layout locations:
        candidates = [
            data_yaml.parent / "manifest.json",
            data_yaml.parent.parent / "manifest.json",
            data_yaml.parent.parent.parent / "manifest.json",
        ]
        for c in candidates:
            if c.is_file():
                manifest_file = c
                break

    manifest_verified = False
    if manifest_file and manifest_file.is_file():
        try:
            m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if "class_map" in m_data:
                m_classes = sorted(m_data["class_map"].items(), key=lambda kv: kv[1])
                expected_names = [name for name, _ in m_classes]
                if expected_names != class_names:
                    errors.append(
                        f"Class names mismatch between manifest ({expected_names}) and data.yaml ({class_names})"
                    )
            if "version_id" not in m_data:
                warnings.append("Dataset manifest is missing version_id")
            manifest_verified = True
        except Exception as exc:
            errors.append(f"Failed to verify dataset manifest at {manifest_file}: {exc}")
    else:
        warnings.append("No dataset manifest (manifest.json) found; provenance cannot be fully audited")

    # 4. Leakage Verification Across Splits
    leakage_free = True
    split_images: dict[str, list[Path]] = {}
    content_hashes_by_split: dict[str, dict[str, Path]] = {}
    groups_by_split: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        if split not in cfg or not cfg[split]:
            continue
        s_img_dir = Path(cfg[split])
        s_img_dir = s_img_dir if s_img_dir.is_absolute() else root / s_img_dir
        if not s_img_dir.is_dir():
            continue
        imgs = sorted(p for p in s_img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
        split_images[split] = imgs
        content_hashes_by_split[split] = {}
        groups_by_split[split] = set()

        for img in imgs:
            # File content hash (exact duplicate check)
            file_hash = _compute_sha256(img)
            content_hashes_by_split[split][file_hash] = img

            # Group / Part ID check
            gid = _extract_group_id(img.name)
            if gid:
                groups_by_split[split].add(gid)

    # Check cross-split content leakage
    splits_present = list(split_images.keys())
    for i in range(len(splits_present)):
        s1 = splits_present[i]
        for j in range(i + 1, len(splits_present)):
            s2 = splits_present[j]

            # A. Duplicate image hash leakage
            common_hashes = set(content_hashes_by_split[s1].keys()) & set(content_hashes_by_split[s2].keys())
            if common_hashes:
                leakage_free = False
                sample_hash = next(iter(common_hashes))
                f1 = content_hashes_by_split[s1][sample_hash].name
                f2 = content_hashes_by_split[s2][sample_hash].name
                errors.append(
                    f"DATA LEAKAGE: Identical image content found in both '{s1}' ({f1}) and '{s2}' ({f2})"
                )

            # B. Group / Part ID leakage
            common_groups = groups_by_split[s1] & groups_by_split[s2]
            if common_groups:
                leakage_free = False
                sample_g = sorted(common_groups)[:3]
                errors.append(
                    f"DATA LEAKAGE: Images from the same group/part {sample_g} exist in both '{s1}' and '{s2}'"
                )

    is_valid = len(errors) == 0
    return HandoverReport(
        valid=is_valid,
        data_yaml=str(data_yaml),
        class_names=class_names,
        splits=splits_info,
        manifest_verified=manifest_verified,
        manifest_path=str(manifest_file) if manifest_file else None,
        leakage_free=leakage_free,
        errors=errors,
        warnings=warnings,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", required=True, type=Path, help="Path to YOLO dataset directory or data.yaml")
    ap.add_argument("--manifest", type=Path, default=None, help="Optional path to manifest.json")
    ap.add_argument("--json", action="store_true", help="Output report in JSON format")
    args = ap.parse_args()

    report = verify_handover(args.yolo_dir, manifest_path=args.manifest)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Handover Verification for: {report.data_yaml}")
        print(f"Class names: {report.class_names}")
        print(f"Manifest Verified: {report.manifest_verified} ({report.manifest_path})")
        print(f"Leakage Free: {report.leakage_free}")
        print(f"Status: {'VALID' if report.valid else 'INVALID'}")
        if report.warnings:
            print("\nWarnings:")
            for w in report.warnings:
                print(f"  - {w}")
        if report.errors:
            print("\nErrors:")
            for e in report.errors:
                print(f"  - {e}")

    return 0 if report.valid else 1


if __name__ == "__main__":
    sys.exit(main())
