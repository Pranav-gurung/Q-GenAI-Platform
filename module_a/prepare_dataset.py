"""End-to-end Module A pipeline: ingest -> validate -> split -> (train-only) augment -> export -> manifest.

Example (synthetic fixture, for software testing only):
  python -m module_a.prepare_dataset --use-fixture --out-root data
Example (real, lawfully obtained dataset):
  python -m module_a.prepare_dataset --image-dir /path/images --annotations /path/ann.csv \
      --classes scratch,crack --source-name "Vendor X" --license-notes "internal use only, contract #..." --out-root data
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .augment_images import AugmentConfig, augment_records
from .exporters import EXPORT_FORMATS, export
from .fixtures import FIXTURE_CLASSES, FIXTURE_NOTICE, generate_fixture
from .ingest import load_annotations
from .schema import Issue, make_class_map, save_canonical
from .split_dataset import SPLITS, check_leakage, split_records
from .validate_dataset import ValidationReport, parse_classes, validate_records
from .version_dataset import build_manifest, compute_version_id, render_readme, save_manifest


class DatasetValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        super().__init__(f"dataset validation failed with {len(report.errors)} error(s); see report")
        self.report = report


def prepare(*, image_dir: Path, annotations: Path, classes: list, out_root: Path, source: dict,
            ratios=(0.70, 0.15, 0.15), seed: int = 42, augment: Optional[AugmentConfig] = None,
            export_formats=("coco",), allow_empty: bool = True, near_dup_threshold: int = 4,
            quarantine_invalid: bool = False, readme_path: Optional[Path] = None) -> dict:
    out_root, image_dir = Path(out_root), Path(image_dir)
    augment = augment or AugmentConfig(seed=seed)
    class_map = make_class_map(classes)

    records, parse_issues = load_annotations(Path(annotations))
    report = validate_records(records, image_dir, classes, allow_empty=allow_empty,
                              near_dup_threshold=near_dup_threshold, parse_issues=parse_issues)
    if not report.valid:
        rep_dir = out_root / "reports" / "failed_validation"
        report.save(rep_dir / "validation.json")
        if not quarantine_invalid:
            raise DatasetValidationError(report)
        # explicit quarantine: invalid images are excluded AND listed in the report
        records = [r for r in records if r.image_id not in report.invalid_image_ids]
    if not records:
        raise DatasetValidationError(report)

    config_blob = {"seed": seed, "ratios": list(ratios), "classes": classes, "augment": augment.to_dict(),
                   "near_dup_threshold": near_dup_threshold, "formats": list(export_formats)}
    version = compute_version_id({r.image_id: r.sha256 for r in records}, config_blob)

    split_stats, split_issues = split_records(records, tuple(ratios), seed, near_dup_threshold)
    leak = check_leakage(records, near_dup_threshold)
    if any(i.severity == "error" for i in leak):
        raise RuntimeError("leakage detected after split: " + "; ".join(i.message for i in leak if i.severity == "error"))
    split_stats["issues"] = [i.to_dict() for i in split_issues]

    splits_root = out_root / "splits" / version
    ann_dir = out_root / "annotations" / version
    proc_dir = out_root / "processed" / version
    rep_dir = out_root / "reports" / version
    for d in (splits_root, ann_dir, proc_dir, rep_dir):
        if d.exists():
            shutil.rmtree(d)  # idempotent rebuild of THIS version only
    by_split = {s: [r for r in records if r.split == s] for s in SPLITS}
    for s, recs in by_split.items():
        (splits_root / s / "images").mkdir(parents=True, exist_ok=True)
        for r in recs:
            shutil.copy2(image_dir / r.file_name, splits_root / s / "images" / Path(r.file_name).name)
            r.file_name = Path(r.file_name).name  # processed copies are flat per split

    # Augment TRAIN ONLY, after splitting
    aug_records, aug_log = [], []
    if augment.variants_per_image > 0:
        aug_records, aug_log = augment_records(by_split["train"], splits_root / "train" / "images",
                                               splits_root / "train" / "images", augment)
    final_records = records + aug_records
    by_split["train"] = by_split["train"] + aug_records
    leak = check_leakage(final_records, near_dup_threshold)
    if any(i.severity == "error" for i in leak):
        raise RuntimeError("leakage detected after augmentation")

    ann_dir = out_root / "annotations" / version
    for s in SPLITS:
        save_canonical(ann_dir / f"canonical_{s}.json", by_split[s], class_map, {"dataset_version": version, "split": s})

    proc_dir = out_root / "processed" / version
    images_dirs = {s: splits_root / s / "images" for s in SPLITS}
    exports = {}
    for fmt in export_formats:
        if fmt != "canonical":
            exports[fmt] = export(fmt, by_split, images_dirs, class_map, proc_dir / fmt)

    rep_dir = out_root / "reports" / version
    report.save(rep_dir / "validation.json")
    (rep_dir / "split_report.json").write_text(json.dumps({"stats": split_stats, "leakage_issues": [i.to_dict() for i in leak]}, indent=2))
    (rep_dir / "augmentation_log.json").write_text(json.dumps(aug_log, indent=2))

    layout = {"splits_dir": str(splits_root), "annotations_dir": str(ann_dir), "processed_dir": str(proc_dir),
              "reports_dir": str(rep_dir), "manifest": str(out_root / "manifests" / f"{version}.json")}
    manifest = build_manifest(
        version_id=version, source=source, records=final_records, class_map=class_map,
        annotation_format=sorted(set(export_formats) - {"canonical"}), split_stats=split_stats,
        augment_config=augment.to_dict(), seed=seed,
        validation_summary={k: v for k, v in report.to_dict().items() if k not in ("issues", "params")},
        outputs=layout,
        extra_limitations=([FIXTURE_NOTICE] if source.get("is_test_fixture") else []))
    save_manifest(manifest, out_root / "manifests")
    readme = render_readme(manifest, layout)
    (rep_dir / "DATASET_README.md").write_text(readme, encoding="utf-8")
    if readme_path:
        Path(readme_path).write_text(readme, encoding="utf-8")
    return {"version": version, "manifest": layout["manifest"], "layout": layout,
            "n_originals": len(records), "n_augmented": len(aug_records),
            "split_warnings": [i.message for i in split_issues]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir")
    ap.add_argument("--annotations")
    ap.add_argument("--classes", help="comma list or file path")
    ap.add_argument("--use-fixture", action="store_true", help="generate a synthetic TEST fixture instead of real data")
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--source-name", default=None)
    ap.add_argument("--license-notes", default=None)
    ap.add_argument("--acquisition-notes", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    ap.add_argument("--variants-per-image", type=int, default=2)
    ap.add_argument("--allow-hflip", action="store_true", help="only if horizontal flips are physically valid")
    ap.add_argument("--allow-vflip", action="store_true")
    ap.add_argument("--formats", default="coco", help=f"comma list of {EXPORT_FORMATS}")
    ap.add_argument("--no-empty", action="store_true", help="treat images without annotations as errors")
    ap.add_argument("--quarantine-invalid", action="store_true", help="exclude (and report) invalid images instead of failing")
    ap.add_argument("--readme-path", default="dataset_README.md")
    a = ap.parse_args(argv)

    out_root = Path(a.out_root)
    if a.use_fixture:
        fx = Path(tempfile.mkdtemp(prefix="fixture_"))
        image_dir, ann = generate_fixture(fx, seed=a.seed)
        classes, is_fixture = FIXTURE_CLASSES, True
        source = {"name": "SYNTHETIC TEST FIXTURE", "path": str(fx), "is_test_fixture": True,
                  "license_notes": "Generated by code; no third-party data.", "acquisition_notes": FIXTURE_NOTICE}
    else:
        if not (a.image_dir and a.annotations and a.classes):
            ap.error("--image-dir, --annotations and --classes are required unless --use-fixture is given")
        image_dir, ann, classes = Path(a.image_dir), Path(a.annotations), parse_classes(a.classes)
        source = {"name": a.source_name or "UNSPECIFIED", "path": str(image_dir), "is_test_fixture": False,
                  "license_notes": a.license_notes, "acquisition_notes": a.acquisition_notes}
    try:
        res = prepare(image_dir=image_dir, annotations=ann, classes=classes, out_root=out_root, source=source,
                      ratios=tuple(float(x) for x in a.ratios.split(",")), seed=a.seed,
                      augment=AugmentConfig(seed=a.seed, variants_per_image=a.variants_per_image,
                                            allow_hflip=a.allow_hflip, allow_vflip=a.allow_vflip),
                      export_formats=tuple(f.strip() for f in a.formats.split(",") if f.strip()),
                      allow_empty=not a.no_empty, quarantine_invalid=a.quarantine_invalid, readme_path=Path(a.readme_path))
    except DatasetValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        for i in e.report.errors[:20]:
            print(f"  [{i.code}] {i.image_id or ''}: {i.message}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
