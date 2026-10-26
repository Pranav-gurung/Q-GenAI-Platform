"""DEMONSTRATION ONLY: simulate an attacker editing an audit record directly in
the database, so you can show that /admin/audit-logs/verify detects it.

    python -m scripts.tamper_demo --list              # show recent events
    python -m scripts.tamper_demo --seq 7             # alter event #7 (backup kept)
    # -> in the UI: Security & audit -> Verify audit chain  => INTEGRITY FAILURE at #7
    python -m scripts.tamper_demo --restore           # put the original value back
    # -> verify again => CHAIN VALID

Never run this against a production database.
"""
import argparse
import json
import sys

from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal
from app.models.audit_event import AuditEvent

BACKUP = get_settings().storage_dir / "tamper_backup.json"


def main() -> int:
    if get_settings().environment == "production":
        print("Refusing to run in production.", file=sys.stderr)
        return 1
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--seq", type=int, help="sequence number of the event to alter")
    g.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    with SessionLocal() as db:
        if args.list:
            for e in db.execute(select(AuditEvent).order_by(AuditEvent.seq.desc()).limit(15)).scalars():
                print(f"#{e.seq:<4} {e.timestamp}  {e.action:28s} {e.metadata_json[:70]}")
            return 0
        if args.restore:
            if not BACKUP.is_file():
                print("No backup found.", file=sys.stderr)
                return 1
            b = json.loads(BACKUP.read_text())
            ev = db.get(AuditEvent, b["seq"])
            ev.metadata_json = b["metadata_json"]
            db.commit()
            BACKUP.unlink()
            print(f"Event #{b['seq']} restored.")
            return 0
        ev = db.get(AuditEvent, args.seq)
        if ev is None:
            print("No such event.", file=sys.stderr)
            return 1
        BACKUP.write_text(json.dumps({"seq": ev.seq, "metadata_json": ev.metadata_json}))
        meta = json.loads(ev.metadata_json)
        meta["tampered"] = True
        if meta.get("ai_decision") == "FAIL":
            meta["ai_decision"] = "PASS"  # "hide" a failed part
        ev.metadata_json = json.dumps(meta, separators=(",", ":"), sort_keys=True)
        db.commit()  # NOTE: the stored hash is left unchanged -> verification must fail
        print(f"Event #{ev.seq} ({ev.action}) altered directly in the database. Backup: {BACKUP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
