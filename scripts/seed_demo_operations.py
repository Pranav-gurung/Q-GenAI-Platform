"""Seed a deterministic SYNTHETIC / DEMO operations dataset for the Operations Intelligence dashboard (Part 7).

    python -m scripts.seed_demo_operations [--dataset data/cv/synthetic-v1] [--end-date 2026-10-18]

Every inspection goes through the CONFIGURED pipeline exactly like a real upload (quality gate -> CV -> rules ->
Part 3 -> RCA / RAG if enabled), using images of the SYNTHETIC CV demo dataset (cartoon scenes, never real
products). Scripted operator reviews are then submitted through the Part 6 feedback service. Only the timestamps are
moved afterwards (to fixed days), so trends are reproducible. Nothing is invented in the results: if the demo model
misses a defect, the dashboard shows that.

Run with CV_ENGINE=yolov8 and a trained demo model for meaningful results (see README "Computer Vision").
Inspections are marked metadata.demo_seed and item ids start with DEMO-OPS-; the dashboard reports how many it
includes. Refuses to run twice on the same database.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
from itertools import cycle
from pathlib import Path

from sqlalchemy import select, update

from backend.core.config import get_settings
from backend.core.exceptions import AppError
from backend.core.logging import configure_logging
from backend.db.models import Inspection, InspectionImage
from backend.db.session import build_engine, build_session_factory
from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.pipeline.registry import build_modules
from backend.schemas.feedback import FeedbackCreate
from backend.schemas.inspection import InspectionCreate
from backend.services.analytics_service import DEMO_SEED_KEY
from backend.services.feedback_service import FeedbackService
from backend.services.inspection_service import InspectionService
from backend.services.reference_service import seed_sample_data
from backend.services.storage import build_storage

SEED_VERSION = "ops-demo-v1"
STAGES = ["RECEIVING", "STORAGE", "PICKING", "PACKING", "DISPATCH"]
CLASS_IDS = {"torn": "1", "leak": "2", "dent": "5", "box": "6", "label": "7"}
STORES = ["DS-BLR-001", "DS-BLR-002", "DS-DEL-001", "DS-MUM-001"]
SKUS = {"leak": ["SKU-MILK-500ML", "SKU-DETERGENT-1L", "SKU-ICECREAM-700ML"],
        "torn": ["SKU-CHIPS-52G", "SKU-BREAD-400G", "SKU-ATTA-5KG"], "dent": ["SKU-COLA-300ML", "SKU-SAUCE-500G"],
        "box": ["SKU-TEA-250G", "SKU-EGGS-6"], "label": ["SKU-SAUCE-500G", "SKU-DETERGENT-1L", "SKU-TEA-250G"],
        None: ["SKU-MILK-500ML", "SKU-CHIPS-52G", "SKU-COLA-300ML", "SKU-EGGS-6"]}
# (day offset from the start, defect or None, stage where the defect image appears, last inspected stage, review)
# review: ok = confirm everything | stage = stage attribution corrected to the previous stage | override = operator
# changes the decision | none = no review. Usefulness ratings follow a fixed pattern.
SCRIPT = [
    (0, "leak", "PACKING", "PACKING", "ok"), (0, None, None, "DISPATCH", "none"), (1, "torn", "PICKING", "PICKING", "ok"),
    (1, "box", "RECEIVING", "RECEIVING", "ok"), (2, "leak", "PACKING", "DISPATCH", "stage"),
    (2, "label", "RECEIVING", "STORAGE", "ok"), (3, None, None, "PACKING", "none"), (3, "dent", "STORAGE", "PICKING", "ok"),
    (4, "leak", "PACKING", "PACKING", "ok"), (4, "torn", "PACKING", "PACKING", "override"),
    (5, None, None, "DISPATCH", "none"), (5, "leak", "DISPATCH", "DISPATCH", "ok"),
    (6, "label", "RECEIVING", "RECEIVING", "ok"), (6, "leak", "PACKING", "PACKING", "stage"),
    (7, "torn", "PICKING", "PACKING", "ok"), (7, None, None, "PICKING", "none"), (8, "leak", "PACKING", "PACKING", "ok"),
    (8, "box", "STORAGE", "STORAGE", "ok"), (9, "leak", "PICKING", "PICKING", "ok"), (9, None, None, "DISPATCH", "none"),
    (10, "leak", "PACKING", "DISPATCH", "ok"), (10, "dent", "PICKING", "PICKING", "override"),
    (11, "label", "RECEIVING", "RECEIVING", "ok"), (11, "leak", "PACKING", "PACKING", "ok"),
    (12, "torn", "PACKING", "PACKING", "stage"), (12, None, None, "DISPATCH", "none"),
    (13, "leak", "PACKING", "PACKING", "ok"), (13, "leak", "PACKING", "DISPATCH", "ok"),
]


def image_pools(root: Path) -> dict[str | None, list[Path]]:
    pools: dict[str | None, list[Path]] = {k: [] for k in [None, *CLASS_IDS]}
    by_id = {v: k for k, v in CLASS_IDS.items()}
    for split in ("test", "val", "train"):
        for lbl in sorted((root / "labels" / split).glob("*.txt")):
            classes = {ln.split()[0] for ln in lbl.read_text().splitlines() if ln.strip()}
            img = root / "images" / split / f"{lbl.stem}.jpg"
            if not img.is_file():
                continue
            if not classes:
                pools[None].append(img)
            elif len(classes) == 1 and next(iter(classes)) in by_id:
                pools[by_id[next(iter(classes))]].append(img)
    return pools


def review_body(kind: str, n: int, stage: str, ai_decision: str | None) -> dict | None:
    useful = {"rca_useful": n % 5 != 2, "action_useful": n % 4 != 1}
    if kind == "ok":
        return {"decision_correct": True, "defect_correct": True, "stage_correct": True, **useful}
    if kind == "stage":
        prev = STAGES[max(STAGES.index(stage) - 1, 0)]
        return {"decision_correct": True, "defect_correct": True, "stage_correct": False, "suspected_stage": prev,
                "comment": f"Defect was already visible at {prev.lower()} (demo review).", **useful}
    if kind == "override":
        if ai_decision == "REVIEW":  # operator settles an uncertain call
            return {"decision_correct": False, "outcome": "FAIL", "comment": "Clearly damaged on inspection (demo).",
                    **useful}
        return {"decision_correct": False, "outcome": "REVIEW", "comment": "Borderline; needs a second look (demo).",
                **useful}
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=Path("data/cv/synthetic-v1"))
    p.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 10, 18))
    args = p.parse_args(argv)
    settings = get_settings()
    configure_logging("WARNING", settings.log_format)
    if not (args.dataset / "labels").is_dir():
        print(f"synthetic dataset not found at {args.dataset} (python -m ml.cv.synthetic --out {args.dataset})")
        return 1
    if settings.cv_engine != "yolov8":
        print(f"warning: CV_ENGINE={settings.cv_engine}; without the YOLO demo model most inspections get no decision")
    pools = {k: iter(v) for k, v in image_pools(args.dataset).items()}
    start = args.end_date - timedelta(days=max(d for d, *_ in SCRIPT))

    engine = build_engine(settings.database_url)
    factory = build_session_factory(engine)
    orchestrator = PipelineOrchestrator(build_modules(settings))
    storage = build_storage(settings)
    stores, sku_cycles = cycle(STORES), {k: cycle(v) for k, v in SKUS.items()}
    summary = {"inspections": 0, "reviews": 0, "skipped_reviews": 0}
    with factory() as db:
        seed_sample_data(db, settings.sample_data_path)
        if db.scalar(select(Inspection.inspection_id).where(Inspection.item_id.like("DEMO-OPS-%")).limit(1)):
            print("demo operations data already present; nothing to do")
            return 0
        insp_svc = InspectionService(db, storage, orchestrator, settings)
        fb_svc = FeedbackService(db, insp_svc)
        for n, (day, defect, at, last, kind) in enumerate(SCRIPT):
            item, store, sku = f"DEMO-OPS-{n + 1:03d}", next(stores), next(sku_cycles[defect])
            when = datetime.combine(start + timedelta(days=day), time(7 + (n % 4) * 2), tzinfo=timezone.utc)
            ids = []
            for stage in STAGES[:STAGES.index(last) + 1]:
                bad = defect is not None and STAGES.index(stage) >= STAGES.index(at)
                img = next(pools[defect if bad else None])
                insp = insp_svc.create(InspectionCreate(store_id=store, sku_id=sku, stage=stage, item_id=item,
                                                        source="API", metadata={DEMO_SEED_KEY: SEED_VERSION}))
                insp_svc.attach_image_and_run(insp.inspection_id, img.read_bytes(), img.name, "image/jpeg")
                ids.append(insp.inspection_id)
            for i, iid in enumerate(ids):  # fixed, ordered timestamps (the only post-hoc change)
                ts = when + timedelta(minutes=40 * i)
                db.execute(update(Inspection).where(Inspection.inspection_id == iid).values(created_at=ts))
                db.execute(update(InspectionImage).where(InspectionImage.inspection_id == iid).values(created_at=ts))
            db.commit()
            summary["inspections"] += len(ids)
            result = insp_svc.get(ids[-1]).result
            body = review_body(kind, n, at or last, ((result.decision or {}) if result else {}).get("outcome"))
            if body:
                try:
                    fb_svc.submit(ids[-1], FeedbackCreate(operator_id=f"OP-DEMO-{n % 3 + 1}", **body))
                    summary["reviews"] += 1
                except AppError:  # e.g. the demo model missed the defect, so there is no stage attribution to review
                    db.rollback()
                    summary["skipped_reviews"] += 1
    engine.dispose()
    print(f"seeded {summary['inspections']} SYNTHETIC demo inspections ({len(SCRIPT)} items, "
          f"{start} .. {args.end_date}); {summary['reviews']} operator reviews, {summary['skipped_reviews']} skipped "
          "(review not applicable to what the model produced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
