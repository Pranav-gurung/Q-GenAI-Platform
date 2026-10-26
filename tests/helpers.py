"""Shared fixtures for Module C tests. All documents here are SYNTHETIC test content, not real standards/SOPs."""
import json
from pathlib import Path

from module_c.config import Settings
from module_c.llm import LLMError

BANNER = "SYNTHETIC TEST DOCUMENT - NOT A REAL STANDARD OR SOP."

HANDLING_P1 = (BANNER + "\n1. SCOPE\nThis test procedure describes handling of machined aluminium housings between stations. "
               "Surface scratches on housings can occur when parts contact each other or hard fixtures during transfer. "
               "Operators shall use separator trays to prevent abrasion of finished surfaces. " * 2)
HANDLING_P2 = ("2. INSPECTION AFTER TRANSFER\nAfter each transfer, inspect housings for scratches and record findings. "
               "If scratches exceed the test acceptance limit, quarantine the lot and notify the quality engineer. "
               "Review tray condition and fixture padding as part of the corrective action. " * 2)
STORAGE_P1 = (BANNER + "\n1. STORAGE CONDITIONS\nCorrosion on steel fasteners is associated with storage in humid environments. "
              "Store fasteners in sealed bags with desiccant and log relative humidity weekly. " * 3)


def write_pdf(path: Path, pages: list) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=A4)
    for text in pages:
        y = 800
        for para in text.split("\n"):
            words, line = para.split(), ""
            for w in words:
                if len(line) + len(w) > 95:
                    c.drawString(40, y, line); y -= 14; line = ""
                line += w + " "
            c.drawString(40, y, line); y -= 14
        c.showPage()
    c.save()
    return path


def write_scanned_pdf(path: Path) -> Path:
    import numpy as np
    from PIL import Image
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200), dtype=np.uint8))
    c = canvas.Canvas(str(path), pagesize=A4)
    c.drawImage(ImageReader(img), 50, 400, 300, 300)
    c.showPage()
    c.save()
    return path


def write_meta(doc_path: Path, document_id: str, title: str, version: str, access_level: str = "internal"):
    (doc_path.with_name(doc_path.name + ".meta.json")).write_text(json.dumps(
        {"document_id": document_id, "title": title, "version": version, "access_level": access_level,
         "license_note": "synthetic test fixture"}))


def make_settings(tmp: Path, **kw) -> Settings:
    base = dict(approved_docs_dir=tmp / "docs", index_dir=tmp / "idx", audit_path=tmp / "audit.jsonl",
                supported_classes=("scratch", "crack", "corrosion"), llm_provider="none", chunk_size=400, chunk_overlap=60)
    base.update(kw)
    return Settings(**base)


class StubProvider:
    name, model, configured = "stub", "stub-model", True

    def __init__(self, replies):
        self.replies, self.prompts, self.calls = list(replies), [], 0

    def generate(self, system, user):
        self.calls += 1
        self.prompts.append(user)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r(user) if callable(r) else r
