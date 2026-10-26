"""Validate a YOLO-format dataset against the defect taxonomy.

    python -m ml.cv.validate_dataset --data data/cv/synthetic-v1/dataset.yaml [--report out.json] [--strict]

Exit code 0 = valid, 1 = errors found (or warnings with --strict), 2 = bad arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.core.config import get_settings
from ml.cv.dataset import validate_dataset
from ml.cv.taxonomy import load_taxonomy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/cv/synthetic-v1/dataset.yaml"))
    parser.add_argument("--taxonomy", type=Path, default=None, help="default: CV_TAXONOMY_PATH")
    parser.add_argument("--report", type=Path, help="write the full report as JSON")
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    parser.add_argument("--max-issues", type=int, default=30, help="issues to print")
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy or get_settings().cv_taxonomy_path)
    report = validate_dataset(args.data, taxonomy)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2))

    print(f"dataset: {report.dataset}")
    for split, n in report.images.items():
        inst = {k: v for k, v in report.instances[split].items() if v}
        print(f"  {split:<5} images={n:<5} negatives={report.negatives[split]:<4} instances={inst}")
    for issue in report.issues[: args.max_issues]:
        print(f"  [{issue.level.upper():7}] {issue.code}: {issue.path}: {issue.message}")
    if len(report.issues) > args.max_issues:
        print(f"  ... {len(report.issues) - args.max_issues} more (use --report)")
    failed = bool(report.errors) or (args.strict and bool(report.warnings))
    print(f"RESULT: {'INVALID' if failed else 'VALID'} ({len(report.errors)} errors, {len(report.warnings)} warnings)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
