"""Export reviewed inspections as JSON Lines learning CANDIDATES for human curation (Part 6).

    python -m scripts.export_learning_candidates [--out data/feedback/candidates.jsonl] [--corrections-only]

Read-only. The file is input for an ML engineer / QA review, not a training set: nothing is retrained here.
data/feedback/ is git-ignored.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.core.config import get_settings
from backend.db.session import build_engine, build_session_factory
from backend.services.learning_service import FeedbackFilters, LearningService


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/feedback/learning_candidates.jsonl"))
    p.add_argument("--corrections-only", action="store_true")
    p.add_argument("--store-id")
    p.add_argument("--limit", type=int, default=100_000)
    args = p.parse_args(argv)
    engine = build_engine(get_settings().database_url)
    with build_session_factory(engine)() as session:
        rows = LearningService(session).candidates(FeedbackFilters(store_id=args.store_id),
                                                   corrections_only=args.corrections_only, limit=args.limit)
    engine.dispose()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} candidate(s) to {args.out} (requires curation before any training use)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
